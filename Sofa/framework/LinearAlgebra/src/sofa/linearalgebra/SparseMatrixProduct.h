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

#include <sofa/linearalgebra/config.h>
#include <utility>
#include <sofa/type/vector.h>
#include <Eigen/Core>

namespace sofa::linearalgebra
{

/**
 * Given two matrices, compute the product of both matrices.
 * The first time the algorithm runs, it computes the "intersection" between both matrices. This intersection can be
 * reused to compute the product. If, later, both input matrices changed their values, but not their sparsity pattern,
 * the "intersection" remains valid. The same intersection can be used to compute the product as long as the sparsity
 * pattern does not change. This strategy is faster than calling the regular matrix product algorithm (but not if the
 * size of the intersection is huge).
 *
 * To compute the product, the method computeProduct must be called.
 *
 * Based on:
 * Saupin, G., Duriez, C. and Grisoni, L., 2007, November. Embedded multigrid approach for real-time volumetric deformation. In International Symposium on Visual Computing (pp. 149-159). Springer, Berlin, Heidelberg.
 * and
 * Saupin, G., 2008. Vers la simulation interactive réaliste de corps déformables virtuels (Doctoral dissertation, Lille 1).
 */
template<class Lhs, class Rhs, class ResultType>
class SparseMatrixProduct
{
public:

    using LhsCleaned = std::decay_t<Lhs>;
    using RhsCleaned = std::decay_t<Rhs>;
    using ResultCleaned = std::decay_t<ResultType>;

    using LhsScalar = typename LhsCleaned::Scalar;
    using RhsScalar = typename RhsCleaned::Scalar;
    using ResultScalar = typename ResultCleaned::Scalar;

    /// Left side of the product A*B
    const LhsCleaned* m_lhs { nullptr };
    /// Right side of the product A*B
    const RhsCleaned* m_rhs { nullptr };

    using Index = Eigen::Index;

    using ProductResult = ResultType;


    void computeProduct(bool forceComputeIntersection = false);
    void computeRegularProduct();

    [[nodiscard]] const ResultType& getProductResult() const { return m_productResult; }

    void invalidateIntersection();

    SparseMatrixProduct(Lhs* lhs, Rhs* rhs) : m_lhs(lhs), m_rhs(rhs) {}
    SparseMatrixProduct() = default;
    virtual ~SparseMatrixProduct() = default;

    struct Intersection
    {
        // Two indices: the first for the values vector of the matrix A, the second for the values vector of the matrix B
        using PairIndex = std::pair<Index, Index>;
        // A list of pairs of indices
        using ListPairIndex = sofa::type::vector<PairIndex>;
        /// Each element of this vector gives the list of values from matrix A and B to multiply together and accumulate
        /// them into the matrix C at the same location in the values vector
        using ValuesIntersection = sofa::type::vector<ListPairIndex>;

        ValuesIntersection intersection;
    };

protected:
    ProductResult m_productResult; /// Result of LHS * RHS

    bool m_hasComputedIntersection { false };
    void computeIntersection();
    virtual void computeProductFromIntersection();

    Intersection m_intersectionAB;

};


}// sofa::linearalgebra
