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
#include <sofa/linearalgebra/FullMatrix.h>

namespace sofa::linearalgebra
{

template<class Real>
FullMatrix<Real>::FullMatrix()
    : data(nullptr), nRow(0), nCol(0), pitch(0), allocsize(0)
{
}

template<class Real>
FullMatrix<Real>::FullMatrix(Index nbRow, Index nbCol)
    : data(new Real[nbRow*nbCol]), nRow(nbRow), nCol(nbCol), pitch(nbCol), allocsize(nbRow*nbCol)
{
}

template<class Real>
FullMatrix<Real>::FullMatrix(Real* p, Index nbRow, Index nbCol)
    : data(p), nRow(nbRow), nCol(nbCol), pitch(nbCol), allocsize(-nbRow*nbCol)
{
}

template<class Real>
FullMatrix<Real>::FullMatrix(Real* p, Index nbRow, Index nbCol, Index pitch)
    : data(p), nRow(nbRow), nCol(nbCol), pitch(pitch), allocsize(-nbRow*pitch)
{
}

template<class Real>
FullMatrix<Real>::~FullMatrix()
{
    if (allocsize>0)
        delete[] data;
}

template<class Real>
typename FullMatrix<Real>::LineIterator FullMatrix<Real>::begin() { return FullMatrix<Real>::LineIterator(data, 0, nCol, pitch); }

template<class Real>
typename FullMatrix<Real>::LineIterator FullMatrix<Real>::end()   { return FullMatrix<Real>::LineIterator(data, nRow, nCol, pitch);   }

template<class Real>
typename FullMatrix<Real>::LineConstIterator FullMatrix<Real>::begin() const { return FullMatrix<Real>::LineConstIterator(data, 0, nCol, pitch); }

template<class Real>
typename FullMatrix<Real>::LineConstIterator FullMatrix<Real>::end()   const { return LineConstIterator(data, nRow, nCol, pitch);   }

template<class Real>
void FullMatrix<Real>::resize(Index nbRow, Index nbCol)
{
    if ( FULLMATRIX_VERBOSE && (nbRow != rowSize() || nbCol != colSize()) )
    {
        msg_info() << /*this->Name() << */": resize(" << nbRow << "," << nbCol << ")";
    }
    if (nbCol != nCol || nbRow != nRow)
    {
        if (allocsize < 0)
        {
            if (nbRow*nbCol > -allocsize)
            {
                msg_error() << "Cannot resize preallocated matrix to size (" << nbRow << "," << nbCol << ")";
                return;
            }
        }
        else
        {
            if (nbRow*nbCol > allocsize)
            {
                if (allocsize > 0)
                    delete[] data;
                allocsize = nbRow*nbCol;
                data = new Real[allocsize];
            }
        }
        pitch = nbCol;
        nCol = nbCol;
        nRow = nbRow;
    }
    clear();
}

template<class Real>
SReal FullMatrix<Real>::element(Index i, Index j) const
{
    if ( FULLMATRIX_CHECK &&  (i >= rowSize() || j >= colSize()) )
    {
        msg_error() << "Invalid read access to element (" << i << "," << j << ") in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return 0.0;
    }
    return (SReal)data[i*pitch+j];
}

template<class Real>
void FullMatrix<Real>::set(Index i, Index j, double v)
{
    msg_info_when(FULLMATRIX_VERBOSE) << /*this->Name() <<*/ "(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") = " << v;
    if ( FULLMATRIX_CHECK &&  (i >= rowSize() || j >= colSize()) )
    {
        msg_error() << "Invalid write access to element (" << i << "," << j << ") in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    data[i*pitch+j] = (Real)v;
}

template<class Real>
void FullMatrix<Real>::add(Index i, Index j, double v)
{
    msg_info_when(FULLMATRIX_VERBOSE) << /*this->Name() << */"(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") += " << v;
    if ( FULLMATRIX_CHECK &&  (i >= rowSize() || j >= colSize()) )
    {
        msg_error() << "Invalid write access to element (" << i << "," << j << ") in "/* << this->Name()*/ << " of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    data[i*pitch+j] += (Real)v;
}

template<class Real>
void FullMatrix<Real>::clear(Index i, Index j)
{
    msg_info_when(FULLMATRIX_VERBOSE) << /*this->Name() <<*/ "(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") = 0";
    if ( FULLMATRIX_CHECK &&  (i >= rowSize() || j >= colSize()) )
    {
        msg_error() << "Invalid write access to element (" << i << "," << j << ") in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    data[i*pitch+j] = (Real)0;
}

template<class Real>
void FullMatrix<Real>::clearRow(Index i)
{
    msg_info_when(FULLMATRIX_VERBOSE) << /*this->Name() <<*/ "(" << rowSize() << "," << colSize() << "): row(" << i << ") = 0";
    if ( FULLMATRIX_CHECK &&  (i >= rowSize()) )
    {
        msg_error() << "Invalid write access to row " << i << " in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    for (Index j=0; j<nCol; ++j)
        data[i*pitch+j] = (Real)0;
}

template<class Real>
void FullMatrix<Real>::clearCol(Index j)
{
    msg_info_when(FULLMATRIX_VERBOSE) <</* this->Name() << */"(" << rowSize() << "," << colSize() << "): col(" << j << ") = 0";
    if ( FULLMATRIX_CHECK &&  (j >= colSize()) )
    {
        msg_error() << "Invalid write access to column " << j << " in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    for (Index i=0; i<nRow; ++i)
        data[i*pitch+j] = (Real)0;
}

template<class Real>
void FullMatrix<Real>::clearRowCol(Index i)
{
    msg_info_when(FULLMATRIX_VERBOSE) << /*this->Name() << */"(" << rowSize() << "," << colSize() << "): row(" << i << ") = 0 and col(" << i << ") = 0";
    if ( FULLMATRIX_CHECK &&  (i >= rowSize() || i >= colSize()) )
    {
        msg_error() << "Invalid write access to row and column " << i << " in " <</*this->Name() << */" of size (" << rowSize() << "," << colSize() << ")";
        return;
    }
    clearRow(i);
    clearCol(i);
}

template<class Real>
void FullMatrix<Real>::clear()
{
    if (pitch == nCol)
        std::fill(data, data+nRow*pitch, (Real)0);
    else
    {
        for (Index i = 0; i<nRow; ++i)
            for (Index j = 0; j<nCol; ++j)
                data[i*pitch + j] = (Real)0;
    }
}

/// matrix-vector product
/// @returns this * v
template<class Real>
FullVector<Real> FullMatrix<Real>::operator*( const FullVector<Real>& v ) const
{
    FullVector<Real> res( rowSize() );
    mul( res, v );
    return res;
}

/// matrix-vector product
/// res = this * v
template<class Real>
void FullMatrix<Real>::mul( FullVector<Real>& res,const FullVector<Real>& b ) const
{
    for( Index i=0 ; i<nRow ; ++i )
    {
        Real r = 0;
        for( Index j=0 ; j<nCol ; ++j )
            r += data[i*pitch+j] * b[j];
        res[i] = r;
    }
}

/// transposed matrix-vector product
/// res = this^T * v
template<class Real>
void FullMatrix<Real>::mulT( FullVector<Real>& res, const FullVector<Real>& b ) const
{
    for( Index i=0 ; i<nCol ; ++i )
    {
        Real r = 0;
        for( Index j=0 ; j<nRow ; ++j )
            r += data[j*pitch+i] * b[j];
        res[i] = r;
    }
}

/// matrix multiplication
/// @returns this * m
template<class Real>
FullMatrix<Real> FullMatrix<Real>::operator*( const FullMatrix<Real>& m ) const
{
    FullMatrix<Real> res( rowSize(), colSize() );
    mul( res, m );
    return res;
}

/// matrix multiplication
/// res = this * m
template<class Real>
void FullMatrix<Real>::mul( FullMatrix<Real>& res, const FullMatrix<Real>& m ) const
{
    assert( m.rowSize() == nCol );

    res.resize( nRow, m.colSize() );
    for( Index i=0 ; i<nRow ; ++i )
    {
        for( unsigned j=0 ; j<(unsigned)m.colSize() ; ++j )
        {
            res.set( i, j, element(i,0)*m.element(0,j) );
            for( Index k=1 ; k<nCol; ++k )
                res.add( i, j, element(i,k)*m.element(k,j) );
        }
    }
}

/// transposed matrix multiplication
/// res = this^T * m
template<class Real>
void FullMatrix<Real>::mulT( FullMatrix<Real>& res, const FullMatrix<Real>& m ) const
{
    assert( m.rowSize() == nRow );

    res.resize( nCol, m.colSize() );
    for( Index i=0 ; i<nCol ; ++i )
    {
        for( unsigned j=0 ; j<(unsigned)m.colSize() ; ++j )
        {
            res.set( i, j, element(0,i)*m.element(0,j) );
            for( Index k=1 ; k<nRow ; ++k )
                res.add( i, j, element(k,i)*m.element(k,j) );
        }
    }
}

template<typename Real>
std::ostream& readFromStream(std::ostream& out, const FullMatrix<Real>& v )
{
    const Index nx = v.colSize();
    const Index ny = v.rowSize();
    out << "[";
    for (Index y=0; y<ny; ++y)
    {
        out << "\n[";
        for (Index x=0; x<nx; ++x)
        {
            out << " " << v.element(y,x);
        }
        out << " ]";
    }
    out << " ]";
    return out;
}

template<typename Real>
LPtrFullMatrix<Real>::LPtrFullMatrix() : ldata(nullptr), lallocsize(0)
{
}

template<typename Real>
LPtrFullMatrix<Real>::~LPtrFullMatrix()
{
    if (lallocsize > 0)
        delete[] ldata;
}

template<typename Real>
void LPtrFullMatrix<Real>::resize(Index nbRow, Index nbCol)
{
    if (nbRow == this->nRow && nbCol == this->nCol)
        this->clear();
    else
    {
        this->FullMatrix<Real>::resize(nbRow, nbCol);
        if (nbRow > lallocsize)
        {
            if (lallocsize > 0)
                delete[] ldata;
            ldata = new Real*[nbRow];
            lallocsize = nbRow;
        }
        for (Index i=0; i<nbRow; ++i)
            ldata[i] = this->data + i*this->pitch;
    }
}

template<>
const char* FullMatrix<double>::Name() { return "FullMatrix"; }

template<>
const char* FullMatrix<float>::Name() { return "FullMatrixf"; }


} // namespace sofa::linearalgebra
