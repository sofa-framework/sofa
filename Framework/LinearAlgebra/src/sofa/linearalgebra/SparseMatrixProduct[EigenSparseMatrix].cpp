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
#define SOFA_LINEARAGEBRA_SPARSEMATRIXPRODUCT_EIGENSPARSEMATRIX_CPP
#include <iostream>
#include <sofa/linearalgebra/SparseMatrixProduct[EigenSparseMatrix].h>
#include <sofa/linearalgebra/SparseMatrixStorageOrder[EigenSparseMatrix].h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::linearalgebra
{

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float> >::computeRegularProduct()
{
    matrixC = (*matrixA) * (*matrixB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double> >::computeRegularProduct()
{
    matrixC = (*matrixA) * (*matrixB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >::computeRegularProduct()
{
    matrixC = (*matrixA) * (*matrixB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >::computeRegularProduct()
{
    matrixC = (*matrixA) * (*matrixB);
}

template <class TMatrix>
void __computeIntersectionColumnMajor(const TMatrix* A, const TMatrix* B, TMatrix* C, typename SparseMatrixProduct<TMatrix>::Intersection& intersection)
{
    C->resize(A->rows(), B->cols());

    *C = (*A) * (*B);

    const SparseMatrixStorageOrder<TMatrix> transpose(A);

    const auto& outerStarts = transpose.getOuterStarts();
    const auto& innerIndices = transpose.getInnerIndices();
    const auto& perm = transpose.getPermutations();

    if (outerStarts.empty())
        return;

    intersection.intersection.clear();
    intersection.intersection.reserve(C->nonZeros());

    for (Eigen::Index c = 0; c < B->outerSize(); ++c)
    {
        const auto beginB = B->outerIndexPtr()[c];
        const auto endB = B->outerIndexPtr()[c + 1];

        for (std::size_t r = 0; r < outerStarts.size() - 1; ++r)
        {
            const auto beginA = outerStarts[r];
            const auto endA = outerStarts[r + 1];

            auto iA = beginA;
            auto iB = beginB;
            typename SparseMatrixProduct<TMatrix>::Intersection::ListPairIndex listPairs;

            while( iA < endA && iB < endB)
            {
                const auto inner_A = innerIndices[iA];
                const auto inner_B = B->innerIndexPtr()[iB];
                if (inner_A == inner_B) //intersection
                {
                    listPairs.emplace_back(perm[iA], iB);
                }

                if (inner_A < inner_B)
                    ++iA;
                else
                    ++iB;
            }

            if (!listPairs.empty())
                intersection.intersection.push_back(listPairs);

        }

    }
}

template <class TMatrix>
void __computeIntersectionRowMajor(const TMatrix* A, const TMatrix* B, TMatrix* C, typename SparseMatrixProduct<TMatrix>::Intersection& intersection)
{
    C->resize(A->rows(), B->cols());

    *C = (*A) * (*B);

    const SparseMatrixStorageOrder<TMatrix> transpose(B);

    const auto& outerStarts = transpose.getOuterStarts();
    const auto& innerIndices = transpose.getInnerIndices();
    const auto& perm = transpose.getPermutations();

    if (outerStarts.empty())
        return;

    intersection.intersection.clear();
    intersection.intersection.reserve(C->nonZeros());

    for (Eigen::Index r = 0; r < A->outerSize(); ++r)
    {
        const auto beginA = A->outerIndexPtr()[r];
        const auto endA = A->outerIndexPtr()[r + 1];

        for (std::size_t c = 0; c < outerStarts.size() - 1; ++c)
        {
            const auto beginB = outerStarts[c];
            const auto endB = outerStarts[c + 1];

            auto iA = beginA;
            auto iB = beginB;
            typename SparseMatrixProduct<TMatrix>::Intersection::ListPairIndex listPairs;

            while( iA < endA && iB < endB)
            {
                const auto inner_A = A->innerIndexPtr()[iA];
                const auto inner_B = innerIndices[iB];
                if (inner_A == inner_B) //intersection
                {
                    listPairs.emplace_back(iA, perm[iB]);
                }

                if (inner_A < inner_B)
                    ++iA;
                else
                    ++iB;
            }

            if (!listPairs.empty())
                intersection.intersection.push_back(listPairs);

        }

    }
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float> >::computeIntersection()
{
    __computeIntersectionColumnMajor(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double> >::computeIntersection()
{
    __computeIntersectionColumnMajor(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >::computeIntersection()
{
    __computeIntersectionRowMajor(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >::computeIntersection()
{
    __computeIntersectionRowMajor(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template<class TMatrix>
void __computeProductFromIntersection(const TMatrix* A, const TMatrix* B, TMatrix* C, const typename SparseMatrixProduct<TMatrix>::Intersection& intersection)
{
    assert(intersection.intersection.size() == C->nonZeros());

    auto* a_ptr = A->valuePtr();
    auto* b_ptr = B->valuePtr();
    auto* c_ptr = C->valuePtr();

    for (const auto& pairs : intersection.intersection)
    {
        auto& value = *c_ptr++;
        value = 0;
        for (const auto& p : pairs)
        {
            value += a_ptr[p.first] * b_ptr[p.second];
        }
    }
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float> >::computeProductFromIntersection()
{
    if (m_hasComputedIntersection == false)
    {
        msg_error("SparseMatrixProduct") << "Intersection computation is required before computing the product";
        return;
    }
    __computeProductFromIntersection(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double> >::computeProductFromIntersection()
{
    if (m_hasComputedIntersection == false)
    {
        msg_error("SparseMatrixProduct") << "Intersection computation is required before computing the product";
        return;
    }
    __computeProductFromIntersection(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >::computeProductFromIntersection()
{
    if (m_hasComputedIntersection == false)
    {
        msg_error("SparseMatrixProduct") << "Intersection computation is required before computing the product";
        return;
    }
    __computeProductFromIntersection(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template <>
void SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >::computeProductFromIntersection()
{
    if (m_hasComputedIntersection == false)
    {
        msg_error("SparseMatrixProduct") << "Intersection computation is required before computing the product";
        return;
    }
    __computeProductFromIntersection(matrixA, matrixB, &matrixC, m_intersectionAB);
}

template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float> >;
template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double> >;

template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >;
template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >;

} //namespace sofa::linearalgebra