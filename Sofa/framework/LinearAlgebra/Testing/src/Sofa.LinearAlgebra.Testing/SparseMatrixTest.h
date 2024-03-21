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
#include <sofa/testing/NumericTest.h>
#include <Eigen/Sparse>

#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::testing
{

template <typename TReal = SReal>
struct SparseMatrixTest : public virtual NumericTest<TReal>
{

    /**
     * Generate a sparse matrix of size nbRows x nbCols. The sparsity is the ratio of non-zero values compared to the
     * total size of the matrix (= nbRows x nbCols).
     */
    template<int EigenSparseMatrixOptions>
    static void generateRandomSparseMatrix(Eigen::SparseMatrix<TReal, EigenSparseMatrixOptions>& eigenMatrix, Eigen::Index nbRows, Eigen::Index nbCols, TReal sparsity)
    {
        if (sparsity < 0 || sparsity > 1)
        {
            msg_error("SparseMatrixTest") << "Invalid sparsity value: " << sparsity << ". Must be between 0 and 1";
            return;
        }

        eigenMatrix.resize(nbRows, nbCols);
        sofa::type::vector<Eigen::Triplet<TReal> > triplets;

        const auto nbNonZero = static_cast<Eigen::Index>(sparsity * static_cast<TReal>(nbRows*nbCols));

        for (Eigen::Index i = 0; i < nbNonZero; ++i)
        {
            const auto value = static_cast<TReal>(sofa::helper::drand(1));
            const auto row = static_cast<Eigen::Index>(sofa::helper::drandpos(nbRows) - 1e-8);
            const auto col = static_cast<Eigen::Index>(sofa::helper::drandpos(nbCols) - 1e-8);
            triplets.emplace_back(row, col, value);
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    template<class InputIt>
    static void generateFromTriplets(Eigen::SparseMatrix<TReal>& eigenMatrix, InputIt first, InputIt last, Eigen::Index nbRows, Eigen::Index nbCols)
    {
        eigenMatrix.resize(nbRows, nbCols);
        eigenMatrix.setFromTriplets(first, last);
    }

    template<
        typename _DstScalar, int _DstOptions, typename _DstStorageIndex,
        typename _SrcScalar, int _SrcOptions, typename _SrcStorageIndex
    >
    static void copyFromEigen(Eigen::SparseMatrix<_DstScalar, _DstOptions, _DstStorageIndex>& dst, const Eigen::SparseMatrix<_SrcScalar, _SrcOptions, _SrcStorageIndex>& src)
    {
        dst = src;
    }

    static void copyFromEigen(linearalgebra::BaseMatrix& dst, const Eigen::SparseMatrix<TReal>& src)
    {
        dst.clear();
        dst.resize(static_cast<linearalgebra::BaseMatrix::Index>(src.rows()), static_cast<linearalgebra::BaseMatrix::Index>(src.cols()));
        for (typename Eigen::SparseMatrix<TReal>::Index k = 0; k < src.outerSize(); ++k)
        {
            for (typename Eigen::SparseMatrix<TReal>::InnerIterator it(src, k); it; ++it)
            {
                dst.add(static_cast<linearalgebra::BaseMatrix::Index>(it.row()),
                    static_cast<linearalgebra::BaseMatrix::Index>(it.col()),
                    it.value());
            }
        }
    }

    template<
        typename _AScalar, int _AOptions, typename _AStorageIndex,
        typename _BScalar, int _BOptions, typename _BStorageIndex
    >
    static bool compareSparseMatrix(const Eigen::SparseMatrix<_AScalar, _AOptions, _AStorageIndex>& A, const Eigen::SparseMatrix<_BScalar, _BOptions, _BStorageIndex>& B)
    {
        return compareEigenSparseMatrix(A, B);
    }

    template<
        typename _AScalar, int _AOptions, typename _AStorageIndex,
        typename _BScalar, int _BOptions, typename _BStorageIndex
    >
    static bool compareEigenSparseMatrix(const Eigen::SparseMatrix<_AScalar, _AOptions, _AStorageIndex>& A, const Eigen::SparseMatrix<_BScalar, _BOptions, _BStorageIndex>& B)
    {
        return A.isApprox(B);
    }


};


} // namespace sofa::testing
