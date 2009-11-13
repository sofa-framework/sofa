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

#include "NewMatMatrix.h"

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


/// Simple full matrix container
template<int LC>
class BlockDiagonalMatrix : public defaulttype::BaseMatrix
{
public:
    typedef double Real;
    typedef int Index;
    typedef NEWMAT::Matrix SubMatrixType;
    typedef NEWMAT::InvertedMatrix InvertedMatrix;

protected:
    std::vector< SubMatrixType > data;
    unsigned cSize;

public:
    int bandWidth;

    BlockDiagonalMatrix()
    {
        bandWidth = LC-1;
    }

    ~BlockDiagonalMatrix() {}

    void resize(int nbRow, int )
    {
        cSize = nbRow;
        data.resize((cSize+LC-1) / LC);
        for (unsigned i=0; i<data.size(); i++) data[i].ReSize(LC,LC);
    }

    unsigned rowSize(void) const
    {
        return cSize;
    }

    unsigned colSize(void) const
    {
        return cSize;
    }

    Real element(int i, int j) const
    {
        if (i/LC!=j/LC) return (Real)0;
        return data[i/LC].element(i%LC,j%LC);
    }

    void set(int i, int j, double v)
    {
        if (i/LC==j/LC) data[i/LC].element(i%LC,j%LC) = (Real)v;
    }

    void add(int i, int j, double v)
    {
        if (i/LC==j/LC) data[i/LC].element(i%LC,j%LC) += (Real)v;
    }

    void clear()
    {
        for (unsigned b=0; b<data.size(); b++)
            for (int j=0; j<LC; j++)
                for (int i=0; i<LC; i++)
                    data[b].element(i,j) = 0.0;;
    }

    void clear(int i, int j)
    {
        if (i/LC==j/LC) data[i/LC].element(i%LC,j%LC) = 0.0;
    }

    void clearRow(int i)
    {
        for (int j=0; j<LC; j++)
            data[i/LC].element(i,j) = 0.0;;
    }

    void clearCol(int j)
    {
        for (int i=0; i<LC; i++)
            data[j/LC].element(i,j) = 0.0;;

    }

    void clearRowCol(int i)
    {
        clearRow(i);
        clearCol(i);
    }

    SubMatrixType sub(int l,int ,int ,int )
    {
        SubMatrixType m=data[l/LC];
        return m;
    }

    void setSubMatrix(unsigned l,unsigned ,unsigned ,unsigned ,InvertedMatrix m)
    {
        data[l/LC] = m;
    }

    void i()
    {
        for (unsigned i=0; i<data.size(); i++) data[i].i();
    }


    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        res.resize(cSize);
        for (unsigned b=0; b<data.size()-1; b++)
        {
            for (int j=0; j<LC; j++)
            {
                res[b*LC+j] = 0;
                for (int i=0; i<LC; i++)
                {
                    res[b*LC+j] += data[b].element(i,j) * v[b*LC+i];
                }
            }
        }

        int last_block = (data.size()-1)*LC;
        for (int j=0; last_block+j<(int) cSize; j++)
        {
            res[last_block+j] = 0;
            for (int i=0; i<LC; i++)
            {
                res[j] += data[data.size()-1].element(i,j) * v[last_block+i];
            }
        }

        return res;
    }

    template<class Real2>
    void mult(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        for (unsigned b=0; b<data.size()-1; b++)
        {
            for (int j=0; j<LC; j++)
            {
                res[b*LC+j] = 0;
                for (int i=0; i<LC; i++)
                {
                    res[b*LC+j] += data[b].element(i,j) * v[b*LC+i];
                }
            }
        }

        int last_block = (data.size()-1)*LC;
        for (int j=0; last_block+j<(int) cSize; j++)
        {
            res[last_block+j] = 0;
            for (int i=0; i<LC; i++)
            {
                res[j] += data[data.size()-1].element(i,j) * v[last_block+i];
            }
        }
    }

    friend std::ostream& operator << (std::ostream& out, const BlockDiagonalMatrix<LC>& v )
    {
        out << "[";
        for (unsigned i=0; i<v.data.size(); i++) out << " " << v.data[i];
        out << " ]";
        return out;
    }

    static const char* Name() { return "BlockDiagonalMatrix"; }
};

class BlockDiagonalMatrix3 : public BlockDiagonalMatrix<3>
{
public :
    static const char* Name() { return "BlockDiagonalMatrix3"; }
};

class BlockDiagonalMatrix6 : public BlockDiagonalMatrix<6>
{
public :
    static const char* Name() { return "BlockDiagonalMatrix6"; }
};

class BlockDiagonalMatrix9 : public BlockDiagonalMatrix<9>
{
public :
    static const char* Name() { return "BlockDiagonalMatrix9"; }
};

class BlockDiagonalMatrix12 : public BlockDiagonalMatrix<12>
{
public :
    static const char* Name() { return "BlockDiagonalMatrix12"; }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
