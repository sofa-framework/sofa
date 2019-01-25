/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_INL

#include "BTDLinearSolver.h"


namespace sofa
{

namespace component
{

namespace linearsolver
{


template<int N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::Bloc::Nrows() const
{
    return BSIZE;
}

template<int N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::Bloc::Ncols() const
{
    return BSIZE;
}

template<int N, typename T>
void  BlocFullMatrix<N, T>::Bloc::resize(Index, Index)
{
    clear();
}

template<int N, typename T>
const T&  BlocFullMatrix<N, T>::Bloc::element(Index i, Index j) const
{
    return (*this)[i][j];
}

template<int N, typename T>
void  BlocFullMatrix<N, T>::Bloc::set(Index i, Index j, const T& v)
{
    (*this)[i][j] = v;
}

template<int N, typename T>
void  BlocFullMatrix<N, T>::Bloc::add(Index i, Index j, const T& v)
{
    (*this)[i][j] += v;
}

template<int N, typename T>
void  BlocFullMatrix<N, T>::Bloc::operator=(const defaulttype::Mat<BSIZE,BSIZE,Real>& v)
{
    defaulttype::Mat<BSIZE,BSIZE,Real>::operator=(v);
}

template<int N, typename T>
defaulttype::Mat<BlocFullMatrix<N,T>::BSIZE,BlocFullMatrix<N,T>::BSIZE,T>  BlocFullMatrix<N, T>::Bloc::operator-() const
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator-();
}

template<int N, typename T>
defaulttype::Mat<BlocFullMatrix<N,T>::BSIZE,BlocFullMatrix<N,T>::BSIZE, T>  BlocFullMatrix<N, T>::Bloc::operator-(const defaulttype::Mat<BSIZE,BSIZE,Real>& m) const
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator-(m);
}

template<int N, typename T>
defaulttype::Vec<BlocFullMatrix<N,T>::BSIZE, T>  BlocFullMatrix<N, T>::Bloc::operator*(const defaulttype::Vec<BSIZE,Real>& v)
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(v);
}

template<int N, typename T>
defaulttype::Mat<BlocFullMatrix<N,T>::BSIZE,BlocFullMatrix<N,T>::BSIZE, T>  BlocFullMatrix<N, T>::Bloc::operator*(const defaulttype::Mat<BSIZE,BSIZE,Real>& m)
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
}

template<int N, typename T>
defaulttype::Mat<BlocFullMatrix<N,T>::BSIZE,BlocFullMatrix<N,T>::BSIZE, T>  BlocFullMatrix<N, T>::Bloc::operator*(const Bloc& m)
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
}

template<int N, typename T>
defaulttype::Mat<BlocFullMatrix<N,T>::BSIZE,BlocFullMatrix<N,T>::BSIZE, T>  BlocFullMatrix<N, T>::Bloc::operator*(const TransposedBloc& mt)
{
    return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(mt.m.transposed());
}

template<int N, typename T>
typename BlocFullMatrix<N,T>::TransposedBloc  BlocFullMatrix<N, T>::Bloc::t() const
{
    return TransposedBloc(*this);
}

template<int N, typename T>
typename BlocFullMatrix<N,T>::Bloc  BlocFullMatrix<N, T>::Bloc::i() const
{
    Bloc r;
    r.invert(*this);
    return r;
}

template<int N, typename T>
typename BlocFullMatrix<N,T>::Index  BlocFullMatrix<N, T>::getSubMatrixDim(Index)
{
    return BSIZE;
}


template<int N, typename T>
BlocFullMatrix<N, T>::BlocFullMatrix()
    : data(NULL), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
{
}

template<int N, typename T>
BlocFullMatrix<N, T>::BlocFullMatrix(Index nbRow, Index nbCol)
    : data(new T[nbRow*nbCol]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize((nbCol/BSIZE)*(nbRow/BSIZE))
{
}

template<int N, typename T>
BlocFullMatrix<N, T>::~BlocFullMatrix()
{
    if (allocsize>0)
        delete[] data;
}

template<int N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::bloc(Index bi, Index bj) const
{
    return data[bi*nBCol + bj];
}

template<int N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::bloc(Index bi, Index bj)
{
    return data[bi*nBCol + bj];
}

template<int N, typename T>
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

template<int N, typename T>
typename BlocFullMatrix<N,T>::Index BlocFullMatrix<N, T>::rowSize(void) const
{
    return nTRow;
}

template<int N, typename T>
typename BlocFullMatrix<N,T>::Index BlocFullMatrix<N, T>::colSize(void) const
{
    return nTCol;
}

template<int N, typename T>
SReal BlocFullMatrix<N, T>::element(Index i, Index j) const
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    return bloc(bi,bj)[i][j];
}

template<int N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::asub(Index bi, Index bj, Index, Index) const
{
    return bloc(bi,bj);
}

template<int N, typename T>
const typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::sub(Index i, Index j, Index, Index) const
{
    return asub(i/BSIZE,j/BSIZE);
}

template<int N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::asub(Index bi, Index bj, Index, Index)
{
    return bloc(bi,bj);
}

template<int N, typename T>
typename BlocFullMatrix<N, T>::Bloc& BlocFullMatrix<N, T>::sub(Index i, Index j, Index, Index)
{
    return asub(i/BSIZE,j/BSIZE);
}

template<int N, typename T>
template<class B>
void BlocFullMatrix<N, T>::getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
{
    m = sub(i,j, nrow, ncol);
}

template<int N, typename T>
template<class B>
void BlocFullMatrix<N, T>::getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
{
    m = asub(bi, bj, nrow, ncol);
}

template<int N, typename T>
template<class B>
void BlocFullMatrix<N, T>::setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
{
    sub(i,j, nrow, ncol) = m;
}

template<int N, typename T>
template<class B>
void BlocFullMatrix<N, T>::setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
{
    asub(bi, bj, nrow, ncol) = m;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::set(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] = (Real)v;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::add(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] += (Real)v;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::clear(Index i, Index j)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    bloc(bi,bj)[i][j] = (Real)0;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::clearRow(Index i)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    for (Index bj = 0; bj < nBCol; ++bj)
        for (Index j=0; j<BSIZE; ++j)
            bloc(bi,bj)[i][j] = (Real)0;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::clearCol(Index j)
{
    Index bj = j / BSIZE; j = j % BSIZE;
    for (Index bi = 0; bi < nBRow; ++bi)
        for (Index i=0; i<BSIZE; ++i)
            bloc(bi,bj)[i][j] = (Real)0;
}

template<int N, typename T>
void BlocFullMatrix<N, T>::clearRowCol(Index i)
{
    clearRow(i);
    clearCol(i);
}

template<int N, typename T>
void BlocFullMatrix<N, T>::clear()
{
    for (Index i=0; i<3*nBRow; ++i)
        data[i].clear();
}

template<int N, typename T>
template<class Real2>
FullVector<Real2> BlocFullMatrix<N, T>::operator*(const FullVector<Real2>& v) const
{
    FullVector<Real2> res(rowSize());
    for (Index bi=0; bi<nBRow; ++bi)
    {
        Index bj = 0;
        for (Index i=0; i<BSIZE; ++i)
        {
            Real r = 0;
            for (Index j=0; j<BSIZE; ++j)
            {
                r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
            }
            res[bi*BSIZE + i] = r;
        }
        for (++bj; bj<nBCol; ++bj)
        {
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index j=0; j<BSIZE; ++j)
                {
                    r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                }
                res[bi*BSIZE + i] += r;
            }
        }
    }
    return res;
}

template<int N, typename T>
BlockVector<N, T>::BlockVector()
{
}

template<int N, typename T>
BlockVector<N, T>::BlockVector(Index n)
    : Inherit(n)
{
}

template<int N, typename T>
BlockVector<N, T>::~BlockVector()
{
}

template<int N, typename T>
typename BlockVector<N, T>::Bloc& BlockVector<N, T>::sub(Index i, Index)
{
    return (Bloc&)*(this->ptr()+i);
}

template<int N, typename T>
const typename BlockVector<N, T>::Bloc& BlockVector<N, T>::asub(Index bi, Index) const
{
    return (const Bloc&)*(this->ptr()+bi*N);
}

template<int N, typename T>
typename BlockVector<N, T>::Bloc& BlockVector<N, T>::asub(Index bi, Index)
{
    return (Bloc&)*(this->ptr()+bi*N);
}

template<int N, typename T>
BTDMatrix<N, T>::BTDMatrix()
    : data(NULL), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
{
}

template<int N, typename T>
BTDMatrix<N, T>::BTDMatrix(Index nbRow, Index nbCol)
    : data(new T[3*(nbRow/BSIZE)]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize(3*(nbRow/BSIZE))
{
}

template<int N, typename T>
BTDMatrix<N, T>::~BTDMatrix()
{
    if (allocsize>0)
        delete[] data;
}

template<int N, typename T>
const typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::bloc(Index bi, Index bj) const
{
    return data[3*bi + (bj - bi + 1)];
}

template<int N, typename T>
typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::bloc(Index bi, Index bj)
{
    return data[3*bi + (bj - bi + 1)];
}

template<int N, typename T>
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

template<int N, typename T>
typename BTDMatrix<N,T>::Index BTDMatrix<N, T>::rowSize(void) const
{
    return nTRow;
}

template<int N, typename T>
typename BTDMatrix<N,T>::Index BTDMatrix<N, T>::colSize(void) const
{
    return nTCol;
}

template<int N, typename T>
SReal BTDMatrix<N, T>::element(Index i, Index j) const
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return (SReal)0;
    return data[bi*3+bindex][i][j];
}

template<int N, typename T>
const typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::asub(Index bi, Index bj, Index, Index) const
{
    static Bloc b;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return b;
    return data[bi*3+bindex];
}

template<int N, typename T>
const typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::sub(Index i, Index j, Index, Index) const
{
    return asub(i/BSIZE,j/BSIZE);
}

template<int N, typename T>
typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::asub(Index bi, Index bj, Index, Index)
{
    static Bloc b;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return b;
    return data[bi*3+bindex];
}

template<int N, typename T>
typename BTDMatrix<N, T>::Bloc& BTDMatrix<N, T>::sub(Index i, Index j, Index, Index)
{
    return asub(i/BSIZE,j/BSIZE);
}

template<int N, typename T>
template<class B>
void BTDMatrix<N, T>::getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
{
    m = sub(i,j, nrow, ncol);
}

template<int N, typename T>
template<class B>
void BTDMatrix<N, T>::getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
{
    m = asub(bi, bj, nrow, ncol);
}

template<int N, typename T>
template<class B>
void BTDMatrix<N, T>::setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
{
    sub(i,j, nrow, ncol) = m;
}

template<int N, typename T>
template<class B>
void BTDMatrix<N, T>::setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
{
    asub(bi, bj, nrow, ncol) = m;
}

template<int N, typename T>
void BTDMatrix<N, T>:: set(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] = (Real)v;
}

template<int N, typename T>
void BTDMatrix<N, T>::add(Index i, Index j, double v)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] += (Real)v;
}

template<int N, typename T>
void BTDMatrix<N, T>::clear(Index i, Index j)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    Index bj = j / BSIZE; j = j % BSIZE;
    Index bindex = bj - bi + 1;
    if (bindex >= 3) return;
    data[bi*3+bindex][i][j] = (Real)0;
}

template<int N, typename T>
void BTDMatrix<N, T>::clearRow(Index i)
{
    Index bi = i / BSIZE; i = i % BSIZE;
    for (Index bj = 0; bj < 3; ++bj)
        for (Index j=0; j<BSIZE; ++j)
            data[bi*3+bj][i][j] = (Real)0;
}

template<int N, typename T>
void BTDMatrix<N, T>::clearCol(Index j)
{
    Index bj = j / BSIZE; j = j % BSIZE;
    if (bj > 0)
        for (Index i=0; i<BSIZE; ++i)
            data[(bj-1)*3+2][i][j] = (Real)0;
    for (Index i=0; i<BSIZE; ++i)
        data[bj*3+1][i][j] = (Real)0;
    if (bj < nBRow-1)
        for (Index i=0; i<BSIZE; ++i)
            data[(bj+1)*3+0][i][j] = (Real)0;
}

template<int N, typename T>
void BTDMatrix<N, T>::clearRowCol(Index i)
{
    clearRow(i);
    clearCol(i);
}

template<int N, typename T>
void BTDMatrix<N, T>::clear()
{
    for (Index i=0; i<3*nBRow; ++i)
        data[i].clear();
}

template<int N, typename T>
template<class Real2>
FullVector<Real2> BTDMatrix<N, T>::operator*(const FullVector<Real2>& v) const
{
    FullVector<Real2> res(rowSize());
    for (Index bi=0; bi<nBRow; ++bi)
    {
        Index b0 = (bi > 0) ? 0 : 1;
        Index b1 = ((bi < nBRow - 1) ? 3 : 2);
        for (Index i=0; i<BSIZE; ++i)
        {
            Real r = 0;
            for (Index bj = b0; bj < b1; ++bj)
            {
                for (Index j=0; j<BSIZE; ++j)
                {
                    r += data[bi*3+bj][i][j] * v[(bi + bj - 1)*BSIZE + j];
                }
            }
            res[bi*BSIZE + i] = r;
        }
    }
    return res;
}




/// Factorize M
///
///     [ A0 C0 0  0  ]         [ a0 0  0  0  ] [ I  l0 0  0  ]
/// M = [ B1 A1 C1 0  ] = L U = [ B1 a1 0  0  ] [ 0  I  l1 0  ]
///     [ 0  B2 A2 C2 ]         [ 0  B2 a2 0  ] [ 0  0  I  l2 ]
///     [ 0  0  B3 A3 ]         [ 0  0  B3 a3 ] [ 0  0  0  I  ]
///     [ a0 a0l0    0       0       ]
/// M = [ B1 B1l0+a1 a1l1    0       ]
///     [ 0  B2      B2l1+a2 a2l2    ]
///     [ 0  0       B3      B3l2+a3 ]
/// L X = [ a0X0 B1X0+a1X1 B2X1+a2X2 B3X2+a3X3 ]
///        [                       inva0                   0             0     0 ]
/// Linv = [               -inva1B1inva0               inva1             0     0 ]
///        [         inva2B2inva1B1inva0       -inva2B2inva1         inva2     0 ]
///        [ -inva3B3inva2B2inva1B1inva0 inva3B3inva2B2inva1 -inva3B3inva2 inva3 ]
/// U X = [ X0+l0X1 X1+l1X2 X2+l2X3 X3 ]
/// Uinv = [ I -l0 l0l1 -l0l1l2 ]
///        [ 0   I  -l1    l1l2 ]
///        [ 0   0    I     -l2 ]
///        [ 0   0    0       I ]
///
///                    [ (I+l0(I+l1(I+l2inva3B3)inva2B2)inva1B1)inva0 -l0(I+l1(I+l2inva3B3)inva2B2)inva1 l0l1(inva2+l2inva3B3inva2) -l0l1l2inva3 ]
/// Minv = Uinv Linv = [    -((I+l1(I+l2inva3B3)inva2B2)inva1B1)inva0    (I+l1(I+l2inva3B3)inva2B2)inva1  -l1(inva2+l2inva3B3inva2)    l1l2inva3 ]
///                    [         (((I+l2inva3B3)inva2B2)inva1B1)inva0       -((I+l2inva3B3)inva2B2)inva1      inva2+l2inva3B3inva2     -l2inva3 ]
///                    [                  -inva3B3inva2B2inva1B1inva0                inva3B3inva2B2inva1             -inva3B3inva2        inva3 ]
///
///                    [ inva0-l0(Minv10)              (-l0)(Minv11)              (-l0)(Minv12)           (-l0)(Minv13) ]
/// Minv = Uinv Linv = [         (Minv11)(-B1inva0) inva1-l1(Minv21)              (-l1)(Minv22)           (-l1)(Minv23) ]
///                    [         (Minv21)(-B1inva0)         (Minv22)(-B2inva1) inva2-l2(Minv32)           (-l2)(Minv33) ]
///                    [         (Minv31)(-B1inva0)         (Minv32)(-B2inva1)         (Minv33)(-B3inva2)       inva3   ]
///
/// if M is symmetric (Ai = Ait and Bi+1 = C1t) :
/// li = invai*Ci = (invai)t*(Bi+1)t = (B(i+1)invai)t
///
///                    [ inva0-l0(Minv11)(-l0t)     Minv10t          Minv20t      Minv30t ]
/// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv22)(-l1t)     Minv21t      Minv31t ]
///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv33)(-l2t) Minv32t ]
///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
///
template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::my_identity(SubMatrix& Id, const Index size_id)
{
    Id.resize(size_id,size_id);
    for (Index i=0; i<size_id; i++)
        Id.set(i,i,1.0);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::invert(SubMatrix& Inv, const BlocType& m)
{
    SubMatrix M;
    M = m;
    // Check for diagonal matrices
    Index i0 = 0;
    const Index n = M.Nrows();
    Inv.resize(n,n);
    while (i0 < n)
    {
        Index j0 = i0+1;
        double eps = M.element(i0,i0)*1.0e-10;
        while (j0 < n)
            if (fabs(M.element(i0,j0)) > eps) break;
            else ++j0;
        if (j0 == n)
        {
            // i0 row is the identity
            Inv.set(i0,i0,(float)1.0/M.element(i0,i0));
            ++i0;
        }
        else break;
    }
    if (i0 < n)
//if (i0 == 0)
        Inv = M.i();
    //else if (i0 < n)
    //        Inv.sub(i0,i0,n-i0,n-i0) = M.sub(i0,i0,n-i0,n-i0).i();
    //else return true;
    //return false;
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::invert(Matrix& M)
{

    msg_info_when(this->f_verbose.getValue()) << "BTDLinearSolver, invert Matrix = "<< M ;

    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    const Index nb = M.rowSize() / bsize;
    if (nb == 0) return;
    //alpha.resize(nb);
    alpha_inv.resize(nb);
    lambda.resize(nb-1);
    B.resize(nb);

    /////////////////////////// subpartSolve init ////////////

    if(subpartSolve.getValue() )
    {
        this->init_partial_inverse(nb,bsize);
    }

    SubMatrix A, C;
    //Index ndiag = 0;
    M.getAlignedSubMatrix(0,0,bsize,bsize,A);
    //if (verbose) sout << "A[0] = " << A << sendl;
    M.getAlignedSubMatrix(0,1,bsize,bsize,C);
    //if (verbose) sout << "C[0] = " << C << sendl;
    //alpha[0] = A;
    invert(alpha_inv[0],A);
    msg_info_when(this->f_verbose.getValue()) << "alpha_inv[0] = " << alpha_inv[0] ;
    lambda[0] = alpha_inv[0]*C;
    msg_info_when(this->f_verbose.getValue()) << "lambda[0] = " << lambda[0] ;

    for (Index i=1; i<nb; ++i)
    {
        M.getAlignedSubMatrix((i  ),(i  ),bsize,bsize,A);
        //if (verbose) sout << "A["<<i<<"] = " << A << sendl;
        M.getAlignedSubMatrix((i  ),(i-1),bsize,bsize,B[i]);
        //if (verbose) sout << "B["<<i<<"] = " << B[i] << sendl;
        //alpha[i] = (A - B[i]*lambda[i-1]);


        BlocType Temp1= B[i]*lambda[i-1];
        BlocType Temp2= A - Temp1;
        invert(alpha_inv[i], Temp2);


        //if(subpartSolve.getValue() ) {
        //	helper::vector<SubMatrix> nHn_1; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
        //	nHn_1.resize(1);
        //	nHn_1[0] = B[i] *alpha_inv[i-1];
        //	H.insert(make_pair(IndexPair(i,i-1),nHn_1[0])); //IndexPair(i+1,i) ??
        //	serr<<" Add pair ("<<i<<","<<i-1<<")"<<sendl;
        //}

        msg_info_when(this->f_verbose.getValue()) << "alpha_inv["<<i<<"] = " << alpha_inv[i] ;
        if (i<nb-1)
        {
            M.getAlignedSubMatrix((i  ),(i+1),bsize,bsize,C);
            lambda[i] = alpha_inv[i]*C;

            msg_info_when(this->f_verbose.getValue()) << "lambda["<<i<<"] = " << lambda[i] ;
        }
    }
    nBlockComputedMinv.resize(nb);
    for (Index i=0; i<nb; ++i)
        nBlockComputedMinv[i] = 0;

    // WARNING : cost of resize here : ???
    Minv.resize(nb*bsize,nb*bsize);
    Minv.setAlignedSubMatrix((nb-1),(nb-1),bsize,bsize,alpha_inv[nb-1]);

    nBlockComputedMinv[nb-1] = 1;

    if(subpartSolve.getValue() )
    {
        SubMatrix iHi; // bizarre: pb compilation avec SubMatrix nHn_1 = B[i] *alpha_inv[i];
        my_identity(iHi, bsize);
        H.insert( make_pair(  IndexPair(nb-1, nb-1), iHi  ) );

        // on calcule les blocks diagonaux jusqu'au bout!!
        // TODO : ajouter un compteur "first_block" qui évite de descendre les déplacements jusqu'au block 0 dans partial_solve si ce block n'a pas été appelé
        computeMinvBlock(0, 0);
    }
}



///
///                    [ inva0-l0(Minv10)     Minv10t          Minv20t      Minv30t ]
/// Minv = Uinv Linv = [  (Minv11)(-l0t)  inva1-l1(Minv21)     Minv21t      Minv31t ]
///                    [  (Minv21)(-l0t)   (Minv22)(-l1t)  inva2-l2(Minv32) Minv32t ]
///                    [  (Minv31)(-l0t)   (Minv32)(-l1t)   (Minv33)(-l2t)   inva3  ]
///

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::computeMinvBlock(Index i, Index j)
{
    if (i < j)
    {
        // i < j correspond to the upper diagonal
        // for the computation, we use the lower diagonal matrix
        Index t = i; i = j; j = t;
    }
    if (nBlockComputedMinv[i] > i-j) return; // the block was already computed



    ///// the block was not computed yet :

    // the block is computed now :
    // 1. all the diagonal block between N and i need to be computed
    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    Index i0 = i;
    while (nBlockComputedMinv[i0]==0)
        ++i0;
    // i0 is the "closest" block of the diagonal that is computed
    // we need to compute all the Minv[i0][i0] (with i0>=i) till i0=i
    while (i0 > i)
    {
        if (nBlockComputedMinv[i0] == 1) // only the bloc on the diagonal is computed : need of the the bloc [i0][i0-1]
        {
            // compute bloc (i0,i0-1)
            //Minv[i0][i0-1] = Minv[i0][i0]*-L[i0-1].t()
            Minv.asub((i0  ),(i0-1),bsize,bsize) = Minv.asub((i0  ),(i0  ),bsize,bsize)*(-(lambda[i0-1].t()));
            ++nBlockComputedMinv[i0];

            if(subpartSolve.getValue() )
            {
                // store -L[i0-1].t() H structure
                SubMatrix iHi_1;
                iHi_1 = - lambda[i0-1].t();
                H.insert( make_pair(  IndexPair(i0, i0-1), iHi_1  ) );
                // compute bloc (i0,i0-1) :  the upper diagonal blocks Minv[i0-1][i0]
                Minv.asub((i0-1),(i0),bsize,bsize) = -lambda[i0-1] * Minv.asub((i0  ),(i0  ),bsize,bsize);
            }

        }


        // compute bloc (i0-1,i0-1)  : //Minv[i0-1][i0-1] = inv(M[i0-1][i0-1]) + L[i0-1] * Minv[i0][i0-1]
        Minv.asub((i0-1),(i0-1),bsize,bsize) = alpha_inv[i0-1] - lambda[i0-1]*Minv.asub((i0  ),(i0-1),bsize,bsize);

        if(subpartSolve.getValue() )
        {
            // store Id in H structure
            SubMatrix iHi;
            my_identity(iHi, bsize);
            H.insert( make_pair(  IndexPair(i0-1, i0-1), iHi  ) );
        }

        ++nBlockComputedMinv[i0-1]; // now Minv[i0-1][i0-1] is computed so   nBlockComputedMinv[i0-1] = 1
        --i0;                       // we can go down to the following block (till we reach i)
    }


    //2. all the block on the lines of block i between the diagonal and the block j are computed
    // i0=i

    Index j0 = i-nBlockComputedMinv[i];


    /////////////// ADD : Calcul pour faire du partial_solve //////////
    // first iHj is initiallized to iHj0+1 (that is supposed to be already computed)
    SubMatrix iHj ;
    if(subpartSolve.getValue() )
    {


        H_it = H.find( IndexPair(i0,j0+1) );
        //serr<<" find pair ("<<i<<","<<j0+1<<")"<<sendl;

        if (H_it == H.end())
        {
            my_identity(iHj, bsize);
            if (i0!=j0+1)
                serr<<"WARNING !! element("<<i0<<","<<j0+1<<") not found : nBlockComputedMinv[i] = "<<nBlockComputedMinv[i]<<sendl;
        }
        else
        {
            //serr<<"element("<<i0<<","<<j0+1<<")  found )!"<<sendl;
            iHj = H_it->second;
        }

    }
    /////////////////////////////////////////////////////////////////////

    while (j0 >= j)
    {
        // compute bloc (i0,j0)
        // Minv[i][j0] = Minv[i][j0+1] * (-L[j0].t)
        Minv.asub((i0  ),(j0  ),bsize,bsize) = Minv.asub((i0  ),(j0+1),bsize,bsize)*(-lambda[j0].t());
        if(subpartSolve.getValue() )
        {
            // iHj0 = iHj0+1 * (-L[j0].t)
            iHj = iHj * -lambda[j0].t();
            H.insert(make_pair(IndexPair(i0,j0),iHj));

            // compute bloc (j0,i0)  the upper diagonal blocks Minv[j0][i0]
            Minv.asub((j0  ),(i0  ),bsize,bsize) = -lambda[j0]*Minv.asub((j0+1),(i0),bsize,bsize);
        }
        ++nBlockComputedMinv[i0];
        --j0;
    }
}

template<class Matrix, class Vector>
double BTDLinearSolver<Matrix,Vector>::getMinvElement(Index i, Index j)
{
    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    if (i < j)
    {
        // lower diagonal
        return getMinvElement(j,i);
    }
    computeMinvBlock(i/bsize, j/bsize);
    return Minv.element(i,j);
}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::solve (Matrix& /*M*/, Vector& x, Vector& b)
{
    msg_info_when(this->f_verbose.getValue() ) << "solve, b = "<< b;

    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    const Index nb = b.size() / bsize;
    if (nb == 0) return;

    x.asub(0,bsize) = alpha_inv[0] * b.asub(0,bsize);
    for (Index i=1; i<nb; ++i)
    {
        x.asub(i,bsize) = alpha_inv[i]*(b.asub(i,bsize) - B[i]*x.asub((i-1),bsize));
    }
    for (Index i=nb-2; i>=0; --i)
    {
        x.asub(i,bsize) /* = Y.asub(i,bsize)- */ -= lambda[i]*x.asub((i+1),bsize);
    }

    // x is the solution of the system
    msg_info_when(this->f_verbose.getValue()) << "solve, solution = "<<x;

}

template<class Matrix, class Vector>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (FullMatrix<float>* r = dynamic_cast<FullMatrix<float>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (defaulttype::BaseMatrix* r = result)
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    return false;
}



///////////////////////////////////////
///////  partial solve  //////////
///////////////////////////////////////


template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::init_partial_inverse(const Index &/*nb*/, const Index &/*bsize*/)
{
    // need to stay in init_partial_inverse (called before inverse)
    H.clear();

}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::init_partial_solve()
{

    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    const Index nb = this->currentGroup->systemRHVector->size() / bsize;

    //TODO => optimisation ??
    bwdContributionOnLH.clear();
    bwdContributionOnLH.resize(nb*bsize);
    fwdContributionOnRH.clear();
    fwdContributionOnRH.resize(nb*bsize);


    _rh_buf.resize(nb*bsize);
    _acc_rh_bloc=0;
    _acc_rh_bloc.resize(bsize);
    _acc_lh_bloc=0;
    _acc_lh_bloc.resize(bsize);

    // Bloc that is currently being proceed => start from the end (so that we use step2 bwdAccumulateLHGlobal and accumulate potential initial forces)
    current_bloc = nb-1;


    // DF represents the variation of the right hand side of the equation (Force in mechanics)
    Vec_dRH.resize(nb);
    for (Index i=0; i<nb; i++)
    {
        Vec_dRH[i]=0;
        Vec_dRH[i].resize(bsize);
        _rh_buf.asub(i,bsize) = this->currentGroup->systemRHVector->asub(i,bsize) ;

    }




}


////// STEP 1


template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::bwdAccumulateRHinBloc(Index indMaxBloc)
{
    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());

    Index b=indMaxBloc;

    //debug
    if (indMaxBloc <  current_bloc)
    {
        std::cout <<" WARNING in bwdAccumulateRHinBloc : indMaxBloc = "<<indMaxBloc <<" <  "<<" current_bloc = "<<current_bloc<<std::endl;
    }

    SubVector RHbloc;
    RHbloc.resize(bsize);

    _acc_lh_bloc= bwdContributionOnLH.asub(b,bsize);


    while(b > current_bloc )
    {

        // evaluate the Right Hand Term for the bloc b
        RHbloc = this->currentGroup->systemRHVector->asub(b,bsize) ;

        // compute the contribution on LH created by RH
        _acc_lh_bloc  += Minv.asub(b,b,bsize,bsize) * RHbloc;

        b--;
        // accumulate this contribution on LH on the lower blocs
        _acc_lh_bloc =  -(lambda[b]*_acc_lh_bloc);

        if (problem.getValue())
            std::cout<<"bwdLH["<<b<<"] = H["<<b<<"]["<<b+1<<"] * ( Minv["<<b+1<<"]["<<b+1<<"] * RH["<<b+1<< "] +bwdLH["<<b+1<<"])"<<std::endl;



        // store the contribution as bwdContributionOnLH
        bwdContributionOnLH.asub(b,bsize) = _acc_lh_bloc;

    }

    b = current_bloc;
    // compute the bloc which indice is current_bloc
    this->currentGroup->systemLHVector->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->currentGroup->systemRHVector->asub(b,bsize) ) +
            bwdContributionOnLH.asub(b, bsize);

    if (problem.getValue())
        std::cout<<"LH["<<b<<"] = Minv["<<b<<"]["<<b<<"] * (fwdRH("<<b<< ") + RH("<<b<<")) + bwdLH("<<b<<")"<<std::endl;


    // here b==current_bloc
}



////// STEP 2

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::bwdAccumulateLHGlobal( )
{
    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    _acc_lh_bloc =  bwdContributionOnLH.asub(current_bloc, bsize);

    while( current_bloc > 0)
    {

        if (problem.getValue())
            std::cout<<"bwdLH["<<current_bloc-1<<"] = H["<<current_bloc-1<<"]["<<current_bloc<<"] *( bwdLH["<<current_bloc<<"] + Minv["<<current_bloc<<"]["<<current_bloc<<"] * RH["<<current_bloc<< "])"<<std::endl;

        // BwdLH += Minv*RH
        _acc_lh_bloc +=  Minv.asub(current_bloc,current_bloc,bsize,bsize) * this->currentGroup->systemRHVector->asub(current_bloc,bsize) ;

        current_bloc--;
        // BwdLH(n-1) = H(n-1)(n)*BwdLH(n)
        _acc_lh_bloc = -(lambda[current_bloc]*_acc_lh_bloc);

        bwdContributionOnLH.asub(current_bloc, bsize) = _acc_lh_bloc;


    }

    // at this point, current_bloc must be equal to 0

    // all the forces from RH were accumulated through bwdAccumulation:
    _indMaxNonNullForce = 0;

    // need to update all the value of LH during forward
    _indMaxFwdLHComputed = 0;

    // init fwdContribution
    fwdContributionOnRH.asub(0, bsize) = 0;


}


/////// STEP 3

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::fwdAccumulateRHGlobal(Index indMinBloc)
{
    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
    _acc_rh_bloc =fwdContributionOnRH.asub(current_bloc, bsize);

    while( current_bloc< indMinBloc)
    {

        // fwdRH(n) += RH(n)
        _acc_rh_bloc += this->currentGroup->systemRHVector->asub(current_bloc,bsize);

        // fwdRH(n+1) = H(n+1)(n) * fwdRH(n)
        _acc_rh_bloc = -(lambda[current_bloc].t() * _acc_rh_bloc);
        current_bloc++;

        fwdContributionOnRH.asub(current_bloc, bsize) = _acc_rh_bloc;

        if (problem.getValue())
            std::cout<<"fwdRH["<<current_bloc<<"] = H["<<current_bloc<<"]["<<current_bloc-1<<"] * (fwdRH["<<current_bloc-1<< "] + RH["<<current_bloc-1<<"])"<<std::endl;

    }

    _indMaxFwdLHComputed = current_bloc;


    Index b = current_bloc;
    // compute the bloc which indice is _indMaxFwdLHComputed
    this->currentGroup->systemLHVector->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->currentGroup->systemRHVector->asub(b,bsize) ) +
            bwdContributionOnLH.asub(b, bsize);

    if (problem.getValue())
        std::cout<<"LH["<<b<<"] = Minv["<<b<<"]["<<b<<"] * (fwdRH("<<b<< ") + RH("<<b<<")) + bwdLH("<<b<<")"<<std::endl;


}


/////// STEP 4

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::fwdComputeLHinBloc(Index indMaxBloc)
{

    const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());

    Index b;

    while(_indMaxFwdLHComputed < indMaxBloc )
    {

        b = _indMaxFwdLHComputed;

        if(b>=0)
        {
            if (problem.getValue())
                std::cout<<" fwdRH["<<b+1<<"] = H["<<b+1<<"]["<<b<<"] * (fwdRH("<<b<< ") + RH("<<b<<"))"<<std::endl;
            // fwdRH(n+1) = H(n+1)(n) * (fwdRH(n) + RH(n))
            fwdContributionOnRH.asub(b+1, bsize) = (-lambda[b].t())* ( fwdContributionOnRH.asub(b, bsize) + this->currentGroup->systemRHVector->asub(b,bsize) ) ;
        }

        _indMaxFwdLHComputed++; b++;

        // compute the bloc which indice is _indMaxFwdLHComputed
        this->currentGroup->systemLHVector->asub(b,bsize) = Minv.asub( b, b ,bsize,bsize) * ( fwdContributionOnRH.asub(b, bsize) + this->currentGroup->systemRHVector->asub(b,bsize) ) +
                bwdContributionOnLH.asub(b, bsize);
        if (problem.getValue())
            std::cout<<"LH["<<b<<"] = Minv["<<b<<"]["<<b<<"] * (fwdRH("<<b<< ") + RH("<<b<<")) + bwdLH("<<b<<")"<<std::endl;

    }




}

template<class Matrix, class Vector>
void BTDLinearSolver<Matrix,Vector>::partial_solve(ListIndex&  Iout, ListIndex&  Iin , bool NewIn)  ///*Matrix& M, Vector& result, Vector& rh, */
{

    Index MinIdBloc_OUT = Iout.front();
    Index MaxIdBloc_OUT = Iout.back();


    //std::cout<<"partial_solve: need update on position for bloc between dofs "<< MinIdBloc_OUT<< "  and "<<MaxIdBloc_OUT<<std::endl;
    if (verification.getValue())
    {
//        const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
//        std::cout<<" input Force= ";
//        for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
//        {
//            std::cout<<"     ["<<i<<"] "<<this->currentGroup->systemRHVector->asub(i,bsize);
//        }
//        std::cout<<" "<<std::endl;
    }


    if( NewIn)
    {

        Index MinIdBloc_IN = Iin.front(); //  Iin needs to be sorted
        Index MaxIdBloc_IN = Iin.back();  //



        //debug
        if (problem.getValue())
            std::cout<<"STEP1: new force on bloc between dofs "<< MinIdBloc_IN<< "  and "<<MaxIdBloc_IN<<std::endl;

        if (MaxIdBloc_IN > this->_indMaxNonNullForce)
            this->_indMaxNonNullForce = MaxIdBloc_IN;

        //step 1:
        bwdAccumulateRHinBloc(this->_indMaxNonNullForce );

        // now the fwdLH begins to be wrong when > to the indice of MinIdBloc_IN (need to be updated in step 3 or 4)
        this->_indMaxFwdLHComputed = MinIdBloc_IN;



    }


    if (current_bloc > MinIdBloc_OUT)
    {
        //debug
        if (problem.getValue())
            std::cout<<"STEP2 (bwd GLOBAL on structure) : current_bloc ="<<current_bloc<<" > to  MinIdBloc_OUT ="<<MinIdBloc_OUT<<std::endl;

        // step 2:
        bwdAccumulateLHGlobal();

        //debug
        if (problem.getValue())
            std::cout<<" new current_bloc = "<<current_bloc<<std::endl;
    }


    if (current_bloc < MinIdBloc_OUT)
    {
        //debug
        if (problem.getValue())
            std::cout<<"STEP3 (fwd GLOBAL on structure) : current_bloc ="<<current_bloc<<" < to  MinIdBloc_OUT ="<<MinIdBloc_OUT<<std::endl;

        //step 3:
        fwdAccumulateRHGlobal(MinIdBloc_OUT);

        // debug
        if (problem.getValue())
            std::cout<<" new current_bloc = "<<current_bloc<<std::endl;
    }



    if ( _indMaxFwdLHComputed < MaxIdBloc_OUT)
    {
        //debug
        if (problem.getValue())
            std::cout<<" STEP 4 :_indMaxFwdLHComputed = "<<_indMaxFwdLHComputed<<" < "<<"MaxIdBloc_OUT = "<<MaxIdBloc_OUT<<"  - verify that current_bloc="<<current_bloc<<" == "<<" MinIdBloc_OUT ="<<MinIdBloc_OUT<<std::endl;

        fwdComputeLHinBloc(MaxIdBloc_OUT );


        //debug
        if (problem.getValue())
            std::cout<<"  new _indMaxFwdLHComputed = "<<_indMaxFwdLHComputed<<std::endl;
    }








    // debug: test
    if (verification.getValue())
    {
        const Index bsize = Matrix::getSubMatrixDim(f_blockSize.getValue());
        Vector *Result_partial_Solve = new Vector();
        (*Result_partial_Solve) = (*this->currentGroup->systemLHVector);

        solve(*this->currentGroup->systemMatrix,*this->currentGroup->systemLHVector, *this->currentGroup->systemRHVector);

        Vector *Result = new Vector();
        (*Result) = (*this->currentGroup->systemLHVector);

        Vector *DR = new Vector();
        (*DR) = (*Result);
        (*DR) -= (*Result_partial_Solve);


        double normDR = 0.0;
        double normR = 0.0;
        for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
        {
            normDR += (DR->asub(i,bsize)).norm();
            normR += (Result->asub(i,bsize)).norm();
        }

        if (normDR > ((1.0e-7)*normR + 1.0e-20) )
        {


            std::cout<<"++++++++++++++++ WARNING +++++++++++\n \n Found solution for bloc OUT :";
            for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
            {
                std::cout<<"     ["<<i<<"] "<< Result_partial_Solve->asub(i,bsize);
            }
            std::cout<<std::endl;

            std::cout<<" after complete resolution OUT :";
            for (Index i=MinIdBloc_OUT; i<=MaxIdBloc_OUT; i++)
            {
                std::cout<<"     ["<<i<<"] "<<Result->asub(i,bsize);
            }
            std::cout<<std::endl;

        }



        delete(Result_partial_Solve);
        delete(Result);
        delete(DR);


        return;
    }



}





template<class Matrix, class Vector>
template<class RMatrix, class JMatrix>
bool BTDLinearSolver<Matrix,Vector>::addJMInvJt(RMatrix& result, JMatrix& J, double fact)
{
    //const Index Jrows = J.rowSize();
    const Index Jcols = J.colSize();
    if (Jcols != Minv.rowSize())
    {
        serr << "BTDLinearSolver::addJMInvJt ERROR: incompatible J matrix size." << sendl;
        return false;
    }


    const bool verbose  = this->f_verbose.getValue();

    if (verbose)
    {
// debug christian: print of the inverse matrix:
        sout<< "C = ["<<sendl;
        for  (Index mr=0; mr<Minv.rowSize(); mr++)
        {
            sout<<" "<<sendl;
            for (Index mc=0; mc<Minv.colSize(); mc++)
            {
                sout<<" "<< getMinvElement(mr,mc);
            }
        }
        sout<< "];"<<sendl;

// debug christian: print of matrix J:
        sout<< "J = ["<<sendl;
        for  (Index jr=0; jr<J.rowSize(); jr++)
        {
            sout<<" "<<sendl;
            for (Index jc=0; jc<J.colSize(); jc++)
            {
                sout<<" "<< J.element(jr, jc) ;
            }
        }
        sout<< "];"<<sendl;
    }


    const typename JMatrix::LineConstIterator jitend = J.end();
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
    {
        Index row1 = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
        {
            Index row2 = jit2->first;
            double acc = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
            {
                Index col1 = i1->first;
                double val1 = i1->second;
                for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                {
                    Index col2 = i2->first;
                    double val2 = i2->second;
                    acc += val1 * getMinvElement(col1,col2) * val2;
                }
            }

            if (verbose)
            {
                sout << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<sendl;
            }

            acc *= fact;
            result.add(row1,row2,acc);
            if (row1!=row2)
                result.add(row2,row1,acc);
        }
    }
    return true;
}

template<> const char* BTDMatrix<1,double>::Name() { return "BTDMatrix1d"; }
template<> const char* BTDMatrix<2,double>::Name() { return "BTDMatrix2d"; }
template<> const char* BTDMatrix<3,double>::Name() { return "BTDMatrix3d"; }
template<> const char* BTDMatrix<4,double>::Name() { return "BTDMatrix4d"; }
template<> const char* BTDMatrix<5,double>::Name() { return "BTDMatrix5d"; }
template<> const char* BTDMatrix<6,double>::Name() { return "BTDMatrix6d"; }


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
