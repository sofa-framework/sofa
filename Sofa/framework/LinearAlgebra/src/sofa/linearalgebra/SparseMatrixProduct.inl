/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/linearalgebra/SparseMatrixProduct.h>
#include <Eigen/Sparse>
#include <sofa/type/vector.h>
#include <numeric>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::linearalgebra::sparsematrixproduct
{

/**
 * Represent a scalar and its index in an array of scalars
 */
template<class Scalar>
struct IndexedValue
{
    Eigen::Index index {};
    Scalar value;

    IndexedValue() = default;

    template<class AnyScalar, typename = std::enable_if_t<std::is_scalar_v<AnyScalar> > >
    IndexedValue(AnyScalar s) : value(s) {}

    IndexedValue(const IndexedValue& other) = default;

    operator Scalar() const
    {
        return value;
    }
};

template<class Scalar>
std::ostream& operator<<(std::ostream& o, IndexedValue<Scalar>& p)
{
    o << "(" << p.value << ", " << p.index << ")";
    return o;
}


/**
 * Represent a sum of scalar products. It stores:
 * - a value for the result
 * - a list of pairs of indices to know what scalars were used for the computation
 */
template<class Scalar>
class IndexValueProduct
{
private:
    using IndexLHS = Eigen::Index;
    using IndexRHS = Eigen::Index;

    using ScalarProduct = std::pair<IndexLHS, IndexRHS>;
    sofa::type::vector<ScalarProduct> m_indices {};

    Scalar value {};

public:

    [[nodiscard]] const sofa::type::vector<ScalarProduct>& getIndices() const
    {
        return m_indices;
    }
    IndexValueProduct() = default;

    template<class AnyScalar, typename = std::enable_if_t<std::is_scalar_v<AnyScalar> > >
    IndexValueProduct(AnyScalar s) : value(s) {}

    operator Scalar() const
    {
        return value;
    }

    template<class AnyScalar>
    IndexValueProduct(const IndexValueProduct<AnyScalar>& other)
        : m_indices(other.indices)
        , value(static_cast<Scalar>(other.value))
    {}

    template<class AnyScalar>
    void operator+=(const IndexValueProduct<AnyScalar>& other)
    {
        m_indices.insert(m_indices.end(), other.m_indices.begin(), other.m_indices.end());
        value += static_cast<Scalar>(other.value);
    }

    template<class ScalarLhs, class ScalarRhs>
    friend IndexValueProduct<decltype(ScalarLhs{} * ScalarRhs{})>
    operator*(const IndexedValue<ScalarLhs>& lhs, const IndexedValue<ScalarRhs>& rhs);
};

template<class Scalar>
std::ostream& operator<<(std::ostream& o, IndexValueProduct<Scalar>& p)
{
    o << "(" << p.value << ", [" << p.indices << "])";
    return o;
}

template<class ScalarLhs, class ScalarRhs>
IndexValueProduct<decltype(ScalarLhs{} * ScalarRhs{})>
operator*(const IndexedValue<ScalarLhs>& lhs, const IndexedValue<ScalarRhs>& rhs)
{
    IndexValueProduct<decltype(ScalarLhs{} * ScalarRhs{})> product;
    product.m_indices.resize(1, {lhs.index, rhs.index});
    product.value = lhs.value * rhs.value;
    return product;
}

}

//this is to inform Eigen that the product of two IndexedValue is a IndexValueProduct
#define DEFINE_PRODUCT_OP_FOR_TYPES(lhs, rhs) \
template<> \
struct Eigen::ScalarBinaryOpTraits< \
    sofa::linearalgebra::sparsematrixproduct::IndexedValue<lhs>, \
    sofa::linearalgebra::sparsematrixproduct::IndexedValue<rhs>, \
    Eigen::internal::scalar_product_op< \
        sofa::linearalgebra::sparsematrixproduct::IndexedValue<lhs>, \
        sofa::linearalgebra::sparsematrixproduct::IndexedValue<rhs> \
    > \
> \
{ \
    typedef sofa::linearalgebra::sparsematrixproduct::IndexValueProduct<decltype(lhs{} * rhs{})> ReturnType; \
};

DEFINE_PRODUCT_OP_FOR_TYPES(float, float)
DEFINE_PRODUCT_OP_FOR_TYPES(double, float)
DEFINE_PRODUCT_OP_FOR_TYPES(float, double)
DEFINE_PRODUCT_OP_FOR_TYPES(double, double)

namespace sofa::linearalgebra
{
template<class Lhs, class Rhs, class ResultType>
void SparseMatrixProduct<Lhs, Rhs, ResultType>::computeProduct(bool forceComputeIntersection)
{
    if (forceComputeIntersection)
    {
        m_hasComputedIntersection = false;
    }

    if (m_hasComputedIntersection == false)
    {
        computeIntersection();
        m_hasComputedIntersection = true;
    }
    else
    {
        computeProductFromIntersection();
    }
}

template <class Lhs, class Rhs, class ResultType>
void SparseMatrixProduct<Lhs, Rhs, ResultType>::computeRegularProduct()
{
    m_productResult = *m_lhs * *m_rhs;
}

template <typename _Scalar, int _Options, typename _StorageIndex>
void flagValueIndices(Eigen::SparseMatrix<sparsematrixproduct::IndexedValue<_Scalar>, _Options, _StorageIndex>& matrix)
{
    for (Eigen::Index i = 0; i < matrix.nonZeros(); ++i)
    {
        matrix.valuePtr()[i].index = i;
    }
}

template<class Lhs, class Rhs, class ResultType>
void SparseMatrixProduct<Lhs, Rhs, ResultType>::computeIntersection()
{
    using LocalLhs = Eigen::SparseMatrix<
        sparsematrixproduct::IndexedValue<LhsScalar>,
        Lhs::Options,
        typename Lhs::StorageIndex
    >;

    using LocalRhs = Eigen::SparseMatrix<
        sparsematrixproduct::IndexedValue<RhsScalar>,
        Rhs::Options,
        typename Rhs::StorageIndex
    >;

    //copy the input matrices in an intermediate matrix with the same properties
    //except that the type of values is IndexedValue
    LocalLhs lhs = m_lhs->template cast<sparsematrixproduct::IndexedValue<LhsScalar>>();
    LocalRhs rhs = m_rhs->template cast<sparsematrixproduct::IndexedValue<RhsScalar>>();

    flagValueIndices(lhs);
    flagValueIndices(rhs);

    using LocalResult = Eigen::SparseMatrix<
        sparsematrixproduct::IndexValueProduct<decltype(LhsScalar{} * RhsScalar{})>,
        ResultType::Options,
        typename ResultType::StorageIndex
    >;

    const LocalResult product = lhs * rhs;

    const auto productNonZeros = product.nonZeros();
    m_intersectionAB.intersection.clear();
    m_intersectionAB.intersection.reserve(productNonZeros);

    for (Eigen::Index i = 0; i < productNonZeros; ++i)
    {
        m_intersectionAB.intersection.push_back(product.valuePtr()[i].getIndices());

        //depending on the storage scheme, Eigen can change the order of the lhs and rhs
        if constexpr (Lhs::IsRowMajor || Rhs::IsRowMajor)
        {
            for (auto& [lhsIndex, rhsIndex] : m_intersectionAB.intersection.back())
            {
                std::swap(lhsIndex, rhsIndex);
            }
        }
    }

    m_productResult = product.template cast<ResultScalar>();
}

template<class Lhs, class Rhs, class ResultType>
void __computeProductFromIntersection(const Lhs* lhs, const Rhs* rhs, ResultType* product,
    const typename SparseMatrixProduct<Lhs, Rhs, ResultType>::Intersection& intersection)
{
    assert(intersection.intersection.size() == product->nonZeros());

    auto* lhs_ptr = lhs->valuePtr();
    auto* rhs_ptr = rhs->valuePtr();
    auto* product_ptr = product->valuePtr();

    for (const auto& pairs : intersection.intersection)
    {
        auto& value = *product_ptr++;
        value = 0;
        for (auto [lhsIndex, rhsIndex] : pairs)
        {
            value += lhs_ptr[lhsIndex] * rhs_ptr[rhsIndex];
        }
    }
}

template<class Lhs, class Rhs, class ResultType>
void SparseMatrixProduct<Lhs, Rhs, ResultType>::computeProductFromIntersection()
{
    if (m_hasComputedIntersection == false)
    {
        msg_error("SparseMatrixProduct") << "Intersection computation is required before computing the product";
        return;
    }
    __computeProductFromIntersection(m_lhs, m_rhs, &m_productResult, m_intersectionAB);
}

template<class Lhs, class Rhs, class ResultType>
void SparseMatrixProduct<Lhs, Rhs, ResultType>::invalidateIntersection()
{
    m_hasComputedIntersection = false;
}

}
