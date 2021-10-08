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

template<class TMatrix>
class SparseMatrixTranspose
{
public:

    using Index = typename TMatrix::Index;

    explicit SparseMatrixTranspose(TMatrix* m)
        : matrix(m)
    {
        buildTranspose();
    }

    void buildTranspose();

    const type::vector<Index>& getOuterStarts() const { return outerStarts; }
    const type::vector<Index>& getInnerIndices() const { return innerIndices; }
    const type::vector<Index>& getPermutations() const { return perm; }

    class InnerIterator;

private:
    /// The matrix to transpose
    TMatrix* matrix { nullptr };

    type::vector<Index> outerStarts;
    type::vector<Index> innerIndices;
    type::vector<Index> perm;
};

template<class TMatrix>
class SparseMatrixTranspose<TMatrix>::InnerIterator
{
public:
    InnerIterator() = delete;
    InnerIterator(const SparseMatrixTranspose<TMatrix>& mat, sofa::Index outer)
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

    InnerIterator& operator+=(const sofa::Index i)
    {
        m_id += i ;
        return *this;
    }

    InnerIterator operator+(Index i)
    {
        InnerIterator result = *this;
        result += i;
        return result;
    }

    SReal value() const;
    typename TMatrix::Index row() const;
    typename TMatrix::Index col() const;

    explicit operator bool() const
    {
        return m_id < m_end;
    }

private:

    const SparseMatrixTranspose<TMatrix>& m_transpose;
    sofa::Index m_id;
    sofa::Index m_end;
    sofa::Index m_outer;
};



} // namespace sofa::linearalgebra