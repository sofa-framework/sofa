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

#include <sofa/type/vector_T.h>

namespace sofa::linearalgebra
{

/**
 * Compute the opposite representation of a sparse matrix.
 * If the input matrix is column-major (resp. row-major), the output matrix is row-major (resp. column-major).
 *
 * The output is not really a matrix per se, but a data structure representing the opposite representation.
 * It also provides a permutation array, that allows to retrieve the initial value of a matrix entry in the input matrix.
 *
 * As mentionned, the output is not a matrix, but a dedicated iterator allows to loop over non-zero entries, in a similar
 * fashion than in Eigen.
 *
 * If only values of the input matrix change, but not its pattern, the output matrix will not change. And the iterator
 * provides the values of the input matrix. It means this data structure can be computed only once as long as the
 * matrix pattern does not change.
 */
template<class TMatrix>
class SparseMatrixStorageOrder
{
public:

    using Index = typename TMatrix::Index;

    explicit SparseMatrixStorageOrder(const TMatrix* m)
        : matrix(m)
    {
        buildOppositeOrder();
    }

    void buildOppositeOrder();

    const type::vector<Index>& getOuterStarts() const { return outerStarts; }
    const type::vector<Index>& getInnerIndices() const { return innerIndices; }
    const type::vector<Index>& getPermutations() const { return perm; }

    class InnerIterator;

private:
    /// The matrix to transpose
    const TMatrix* matrix { nullptr };

    type::vector<Index> outerStarts;
    type::vector<Index> innerIndices;
    type::vector<Index> perm;
};

template<class TMatrix>
class SparseMatrixStorageOrder<TMatrix>::InnerIterator
{
public:
    InnerIterator() = delete;
    InnerIterator(const SparseMatrixStorageOrder<TMatrix>& mat, sofa::Index outer)
        : m_transpose(mat)
        , m_outer(outer)
    {
        m_id = m_transpose.outerStarts[outer];
        m_end = m_transpose.outerStarts[outer + 1];
    }

    InnerIterator& operator++()
    {
        ++m_id;
        return *this;
    }

    InnerIterator& operator+=(const typename TMatrix::Index i)
    {
        m_id += i ;
        return *this;
    }

    InnerIterator operator+(const typename TMatrix::Index i)
    {
        InnerIterator result = *this;
        result += i;
        return result;
    }

    typename TMatrix::Scalar value() const;
    typename TMatrix::Index row() const;
    typename TMatrix::Index col() const;

    explicit operator bool() const
    {
        return m_id < m_end;
    }

private:

    const SparseMatrixStorageOrder<TMatrix>& m_transpose;
    typename TMatrix::Index m_id;
    typename TMatrix::Index m_end;
    typename TMatrix::Index m_outer;
};



} // namespace sofa::linearalgebra