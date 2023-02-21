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
#define SOFA_LINEARAGEBRA_SPARSEMATRIXTRANSPOSE_EIGENSPARSEMATRIX_CPP
#include <iostream>
#include <sofa/linearalgebra/SparseMatrixStorageOrder[EigenSparseMatrix].h>

namespace sofa::linearalgebra
{

template<class TMatrix>
void transpose(const TMatrix& self, type::vector<typename TMatrix::Index>& outer, type::vector<typename TMatrix::Index>& inner, type::vector<typename TMatrix::Index>& perm)
{
    const auto aOuterSize = self.innerSize();
    const auto nbNonZero = self.nonZeros();

    outer.clear();
    inner.clear();

    //will be shifted twice
    outer.resize(aOuterSize + 2, 0);

    for(Eigen::Index i = 0; i < nbNonZero; ++i)
    {
        const auto innerId = self.innerIndexPtr()[i];
        //count how many non-zero values are in the row rowId
        ++outer[innerId+2];
    }

    //build outer: for each row the index of the first non-zero
    //It's the first shift
    for(std::size_t i = 2; i < outer.size(); ++i)
    {
        const auto nbNonzeroInRow = outer[i];
        const auto previousRowId = outer[i-1];
        outer[i] = previousRowId + nbNonzeroInRow;
    }

    inner.resize(nbNonZero);
    perm.resize(nbNonZero);

    for (Eigen::Index selfOuterId = 0; selfOuterId < self.outerSize(); ++selfOuterId)
    {
        for (auto aId = self.outerIndexPtr()[selfOuterId]; aId < self.outerIndexPtr()[selfOuterId + 1]; ++aId)
        {
            const auto rowA = self.innerIndexPtr()[aId];

            const auto newIndex = outer[rowA + 1]++; //second shift
            perm[newIndex] = aId;
            inner[newIndex] = selfOuterId;
        }
    }

    outer.resize(aOuterSize + 1);
}

template<>
void SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::buildOppositeOrder()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
float SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::row() const
{
    return m_outer;
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::col() const
{
    return m_transpose.innerIndices[m_id];
}

template<>
void SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::buildOppositeOrder()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
double SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::row() const
{
    return m_outer;
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::col() const
{
    return m_transpose.innerIndices[m_id];
}


template<>
void SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::buildOppositeOrder()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
float SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<float, Eigen::RowMajor>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::row() const
{
    return m_transpose.innerIndices[m_id];
}

template<>
Eigen::SparseMatrix<float, Eigen::RowMajor>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::col() const
{
    return m_outer;
}

template<>
void SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::buildOppositeOrder()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
double SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<double, Eigen::RowMajor>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::row() const
{
    return m_transpose.innerIndices[m_id];
}

template<>
Eigen::SparseMatrix<double, Eigen::RowMajor>::Index SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::col() const
{

    return m_outer;
}


template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >;
template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >;

template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >;
template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >;

} //namespace sofa::linearalgebra
