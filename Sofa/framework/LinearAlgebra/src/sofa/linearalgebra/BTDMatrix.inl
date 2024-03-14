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

#include <sofa/linearalgebra/BTDMatrix.h>

namespace sofa::linearalgebra
{

template<std::size_t N, typename T>
BTDMatrix<N, T>::BTDMatrix()
    : data(nullptr), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
{
}

template<std::size_t N, typename T>
BTDMatrix<N, T>::BTDMatrix(Index nbRow, Index nbCol)
    : data(new Block[3*(nbRow/BSIZE)]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize(3*(nbRow/BSIZE))
{
}

template<std::size_t N, typename T>
BTDMatrix<N, T>::~BTDMatrix()
{
    if (allocsize>0)
        delete[] data;
}

template<std::size_t N, typename T>
const typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::bloc(Index bi, Index bj) const
{
    return data[3*bi + (bj - bi + 1)];
}

template<std::size_t N, typename T>
typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::bloc(Index bi, Index bj)
{
    return data[3*bi + (bj - bi + 1)];
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::resize(Index nbRow, Index nbCol)
{
    if (nbCol != nTCol || nbRow != nTRow)
    {
        if (allocsize < 0)
        {
            if ((nbRow/BSIZE)*3 > -allocsize)
            {
                msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size ("<<nbRow<<","<<nbCol<<")" ;
                return;
            }
        }
        else
        {
            if ((nbRow/BSIZE)*3 > allocsize)
            {
                if (allocsize > 0)
                    delete[] data;
                allocsize = (nbRow/BSIZE)*3;
                data = new Block[allocsize];
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
typename BTDMatrix<N,T>::Index BTDMatrix<N, T>::rowSize(void) const
{
    return nTRow;
}

template<std::size_t N, typename T>
typename BTDMatrix<N,T>::Index BTDMatrix<N, T>::colSize(void) const
{
    return nTCol;
}

template<std::size_t N, typename T>
SReal BTDMatrix<N, T>::element(Index i, Index j) const
{
    const Index bi = i / BSIZE; i = i % BSIZE;
    const Index bj = j / BSIZE; j = j % BSIZE;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return (SReal)0;
    return data[bi*3+bindex][i][j];
}

template<std::size_t N, typename T>
const typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::asub(Index bi, Index bj, Index, Index) const
{
    static Block b;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return b;
    return data[bi*3+bindex];
}

template<std::size_t N, typename T>
const typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::sub(Index i, Index j, Index k, Index l) const
{
    return asub(i/BSIZE,j/BSIZE,k,l);
}

template<std::size_t N, typename T>
typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::asub(Index bi, Index bj, Index, Index)
{
    static Block b;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return b;
    return data[bi*3+bindex];
}

template<std::size_t N, typename T>
typename BTDMatrix<N, T>::Block& BTDMatrix<N, T>::sub(Index i, Index j, Index k, Index l)
{
    return asub(i/BSIZE,j/BSIZE,k,l);
}

template<std::size_t N, typename T>
template<class B>
void BTDMatrix<N, T>::getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
{
    m = sub(i,j, nrow, ncol);
}

template<std::size_t N, typename T>
template<class B>
void BTDMatrix<N, T>::getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
{
    m = asub(bi, bj, nrow, ncol);
}

template<std::size_t N, typename T>
template<class B>
void BTDMatrix<N, T>::setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
{
    sub(i,j, nrow, ncol) = m;
}

template<std::size_t N, typename T>
template<class B>
void BTDMatrix<N, T>::setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
{
    asub(bi, bj, nrow, ncol) = m;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>:: set(Index i, Index j, double v)
{
    const Index bi = i / BSIZE; i = i % BSIZE;
    const Index bj = j / BSIZE; j = j % BSIZE;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] = (Real)v;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::add(Index i, Index j, double v)
{
    const Index bi = i / BSIZE; i = i % BSIZE;
    const Index bj = j / BSIZE; j = j % BSIZE;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] += (Real)v;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::clear(Index i, Index j)
{
    const Index bi = i / BSIZE; i = i % BSIZE;
    const Index bj = j / BSIZE; j = j % BSIZE;
    const Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::clearRow(Index i)
{
    const Index bi = i / BSIZE; i = i % BSIZE;
    for (Index bj = 0; bj < 3; ++bj)
        for (Index j=0; j<BSIZE; ++j)
            data[bi*3+bj][i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::clearCol(Index j)
{
    const Index bj = j / BSIZE; j = j % BSIZE;
    if (bj > 0)
        for (Index i=0; i<BSIZE; ++i)
            data[(bj-1)*3+2][i][j] = (Real)0;
    for (Index i=0; i<BSIZE; ++i)
        data[bj*3+1][i][j] = (Real)0;
    if (bj < nBRow-1)
        for (Index i=0; i<BSIZE; ++i)
            data[(bj+1)*3+0][i][j] = (Real)0;
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::clearRowCol(Index i)
{
    clearRow(i);
    clearCol(i);
}

template<std::size_t N, typename T>
void BTDMatrix<N, T>::clear()
{
    for (Index i=0; i<3*nBRow; ++i)
        data[i].clear();
}

} // namespace sofa::linearalgebra
