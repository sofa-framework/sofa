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
#include <sofa/linearalgebra/BlocFullMatrix.h>

namespace sofa::linearalgebra
{

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::Bloc::Nrows() const
{
    return BSIZE;
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::Bloc::Ncols() const
{
    return BSIZE;
}

template<std::size_t N, typename T>
void  BlocFullMatrix<N, T>::Bloc::resize(Index, Index)
{
    clear();
}

template<std::size_t N, typename T>
const T&  BlocFullMatrix<N, T>::Bloc::element(Index i, Index j) const
{
    return (*this)[i][j];
}

template<std::size_t N, typename T>
void  BlocFullMatrix<N, T>::Bloc::set(Index i, Index j, const T& v)
{
    (*this)[i][j] = v;
}

template<std::size_t N, typename T>
void  BlocFullMatrix<N, T>::Bloc::add(Index i, Index j, const T& v)
{
    (*this)[i][j] += v;
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::TransposedBloc  BlocFullMatrix<N, T>::Bloc::t() const
{
    return TransposedBloc(*this);
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Bloc  BlocFullMatrix<N, T>::Bloc::i() const
{
    Bloc r;
    r.invert(*this);
    return r;
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::getSubMatrixDim(Index)
{
    return BSIZE;
}


template<std::size_t N, typename T>
BlocFullMatrix<N, T>::BlocFullMatrix()
    : data(nullptr), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
{
}

template<std::size_t N, typename T>
BlocFullMatrix<N, T>::BlocFullMatrix(Index nbRow, Index nbCol)
    : data(new T[nbRow*nbCol]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize((nbCol/BSIZE)*(nbRow/BSIZE))
{
}

template<std::size_t N, typename T>
BlocFullMatrix<N, T>::~BlocFullMatrix()
{
    if (allocsize>0)
        delete[] data;
}

template<std::size_t N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::bloc(Index bi, Index bj) const
{
    return data[bi*nBCol + bj];
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::bloc(Index bi, Index bj)
{
    return data[bi*nBCol + bj];
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::resize(Index nbRow, Index nbCol)
{
    if (nbCol != nTCol || nbRow != nTRow)
    {
        if (allocsize < 0)
        {
            if ((nbCol/BSIZE)*(nbRow/BSIZE) > -allocsize)
            {
                msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size ("<<nbRow<<","<<nbCol<<")." ;
                return;
            }
        }
        else
        {
            if ((nbCol/BSIZE)*(nbRow/BSIZE) > allocsize)
            {
                if (allocsize > 0)
                    delete[] data;
                allocsize = (nbCol/BSIZE)*(nbRow/BSIZE);
                data = new Bloc[allocsize];
            }
        }
        nTCol = nbCol;
        nTRow = nbRow;
        nBCol = nbCol/BSIZE;
        nBRow = nbRow/BSIZE;
    }
    clear();
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Index BlocFullMatrix<N, T>::rowSize(void) const
{
    return nTRow;
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N,T>::Index BlocFullMatrix<N, T>::colSize(void) const
{
    return nTCol;
}

template<std::size_t N, typename T>
SReal BlocFullMatrix<N, T>::element(Index i, Index j) const
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    return bloc(bi,bj)[i][j];
}

template<std::size_t N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::asub(Index bi, Index bj, Index, Index) const
{
    return bloc(bi,bj);
}

template<std::size_t N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::sub(Index i, Index j, Index, Index) const
{
    return asub(i/BSIZE,j/BSIZE);
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::asub(Index bi, Index bj, Index, Index)
{
    return bloc(bi,bj);
}

template<std::size_t N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::sub(Index i, Index j, Index, Index)
{
    return asub(i/BSIZE,j/BSIZE);
}

template<std::size_t N, typename T>
template<class B>
void BlocFullMatrix<N, T>::getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
{
    m = sub(i,j, nrow, ncol);
}

template<std::size_t N, typename T>
template<class B>
void BlocFullMatrix<N, T>::getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
{
    m = asub(bi, bj, nrow, ncol);
}

template<std::size_t N, typename T>
template<class B>
void BlocFullMatrix<N, T>::setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
{
    sub(i,j, nrow, ncol) = m;
}

template<std::size_t N, typename T>
template<class B>
void BlocFullMatrix<N, T>::setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
{
    asub(bi, bj, nrow, ncol) = m;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::set(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] = (Real)v;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::add(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] += (Real)v;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::clear(Index i, Index j)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::clearRow(Index i)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    for (Index bj = 0; bj < nBCol; ++bj)
        for (Index j=0; j<BSIZE; ++j)
            bloc(bi,bj)[i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::clearCol(Index j)
{
    Index bj = j / BSIZE; j = j % BSIZE;
    for (Index bi = 0; bi < nBRow; ++bi)
        for (Index i=0; i<BSIZE; ++i)
            bloc(bi,bj)[i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::clearRowCol(Index i)
{
    clearRow(i);
    clearCol(i);
}

template<std::size_t N, typename T>
void BlocFullMatrix<N, T>::clear()
{
    for (Index i=0; i<3*nBRow; ++i)
        data[i].clear();
}


} // namespace sofa::linearalgebra
