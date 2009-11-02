/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_DIAGONALMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_DIAGONALMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include "FullVector.h"

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Simple full matrix container
template<typename T>
class DiagonalMatrix : public defaulttype::BaseMatrix
{
public:
    typedef T Real;
    typedef int Index;

protected:
    FullVector<T> data;

public:

    DiagonalMatrix()
        : data(NULL)
    {
    }

    DiagonalMatrix(int nbRow, int nbCol)
        : data(new T[nbRow])
    {
    }

    DiagonalMatrix(Real* p, int nbRow, int nbCol)
        : data(p)
    {
    }

    ~DiagonalMatrix() {}

    Real* ptr() { return data.ptr(); }
    const Real* ptr() const { return data.ptr(); }

    Real* operator[](Index i)
    {
        return data+i;
    }

    const Real* operator[](Index i) const
    {
        return data+i;
    }

    void resize(int nbRow, int /*nbCol*/)
    {
        data.resize(nbRow);
    }

    unsigned int rowSize(void) const
    {
        return data.size();
    }

    unsigned int colSize(void) const
    {
        return data.size();
    }

    SReal element(int i, int j) const
    {
        if (i!=j) return (Real)0;
        return data[i];
    }

    void set(int i, int j, double v)
    {
        if (i==j) data[i] = (Real)v;
    }

    void add(int i, int j, double v)
    {
        if (i==j) data[i] += (Real)v;
    }

    void clear(int i, int /*j*/)
    {
        data[i] = (Real)0;
    }

    void clearRow(int i)
    {
        data[i] = (Real)0;
    }

    void clearCol(int j)
    {
        data[j] = (Real)0;
    }

    void clearRowCol(int i)
    {
        data[i] = (Real)0;
    }

    void clear()
    {
        data.clear();
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        for (int i=0; i<rowSize(); i++) res[i] = data[i] * v[i];
        return res;
    }

    friend std::ostream& operator << (std::ostream& out, const DiagonalMatrix<T>& v )
    {
        int ny = v.rowSize();
        out << "[";
        for (int y=0; y<ny; ++y) out << " " << v.element(y);
        out << " ]";
        return out;
    }

    static const char* Name() { return "DiagonalMatrix"; }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
