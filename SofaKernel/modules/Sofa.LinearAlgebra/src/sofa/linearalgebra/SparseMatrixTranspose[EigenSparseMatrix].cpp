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
#include <sofa/linearalgebra/SparseMatrixTranspose[EigenSparseMatrix].h>

namespace sofa::linearalgebra
{

template<class TMatrix>
void transpose(const TMatrix& self, type::vector<typename TMatrix::Index>& outer, type::vector<typename TMatrix::Index>& inner, type::vector<typename TMatrix::Index>& perm)
{
    const auto aCols = self.cols();
    const auto nbNonZero = self.nonZeros();

    outer.clear();
    inner.clear();

    //will be shifted twice
    outer.resize(aCols + 2, 0);
    
    for(Eigen::Index i = 0; i < nbNonZero; ++i)
    {
        const auto rowId = self.innerIndexPtr()[i];
        //count how many non-zero values are in the row rowId
        ++outer[rowId+2];
    }
    
    //build outer: for each row the index of the first non-zero
    //It's the first shift
    for(Eigen::Index i = 2; i < aCols + 1; ++i)
    {
        const auto nbNonzeroInRow = outer[i];
        const auto previousRowId = outer[i-1];
        outer[i] = previousRowId + nbNonzeroInRow;
    }

    inner.resize(nbNonZero);
    perm.resize(nbNonZero);
    
    for (Eigen::Index i = 0; i < aCols; ++i)
    {
        const auto colA = i;
        for (auto aId = self.outerIndexPtr()[i]; aId < self.outerIndexPtr()[i+1]; ++aId)
        {
            //aId is the index of the first non-zero value in the column i of matrix A
            const auto rowA = self.innerIndexPtr()[aId];

            const auto newIndex = outer[rowA + 1]++; //second shift
            perm[newIndex] = aId;
            inner[newIndex] = colA;
        }
    }

    outer.resize(aCols + 1);
}

template<>
void SparseMatrixTranspose<Eigen::SparseMatrix<float> >::buildTranspose()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
SReal SparseMatrixTranspose<Eigen::SparseMatrix<float> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixTranspose<Eigen::SparseMatrix<float> >::InnerIterator::row() const
{
    return m_transpose.innerIndices[m_id];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixTranspose<Eigen::SparseMatrix<float> >::InnerIterator::col() const
{
    return m_outer;
}

template<>
void SparseMatrixTranspose<Eigen::SparseMatrix<double> >::buildTranspose()
{
    transpose(*matrix, outerStarts, innerIndices, perm);
}

template<>
SReal SparseMatrixTranspose<Eigen::SparseMatrix<double> >::InnerIterator::value() const
{
    return m_transpose.matrix->valuePtr()[m_transpose.perm[m_id]];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixTranspose<Eigen::SparseMatrix<double> >::InnerIterator::row() const
{
    return m_transpose.innerIndices[m_id];
}

template<>
Eigen::SparseMatrix<float>::Index SparseMatrixTranspose<Eigen::SparseMatrix<double> >::InnerIterator::col() const
{
    return m_outer;
}

template class SOFA_LINEARALGEBRA_API SparseMatrixTranspose<Eigen::SparseMatrix<float> >;
template class SOFA_LINEARALGEBRA_API SparseMatrixTranspose<Eigen::SparseMatrix<double> >;

} //namespace sofa::linearalgebra