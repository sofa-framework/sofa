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
#ifndef SOFA_GPU_CUDA_CUDATYPES_BASE_H
#define SOFA_GPU_CUDA_CUDATYPES_BASE_H

#include "CudaTypes.h"
#include <sofa/component/linearsolver/FullMatrix.h>

//#define DEBUG_BASE

namespace sofa
{
namespace gpu
{
namespace cuda
{

using namespace sofa::defaulttype;

template <class T>
class CudaBaseMatrix : public BaseMatrix
{
public :

    CudaMatrix<T> & getCudaMatrix()
    {
        return m;
    }

    void resize(int nbRow, int nbCol)
    {
        m.resize(nbCol,nbRow,BSIZE);
    }

    void resize(int nbRow, int nbCol,int ws)
    {
        m.resize(nbCol,nbRow,ws);
    }

    unsigned int rowSize() const
    {
        return m.getSizeY();
    }

    unsigned int colSize() const
    {
        return m.getSizeX();
    }

    SReal element(int i, int j) const
    {
        return m[i][j];
    }

    const T* operator[] ( int i ) const
    {
        return m[i];
    }

    void clear()
    {
// 			for (unsigned j=0; j<m.getSizeX(); j++) {
// 				for (unsigned i=0; i<m.getSizeY(); i++) {
// 				  m[j][i] = (T)(0.0);
// 				}
// 			}
        m.memsetHost();
    }

    void set(int i, int j, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[i][j] = (T)v;
    }

    void add(int i, int j, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[i][j] += (T)v;
    }

    static std::string Name()
    {
        return "CudaBaseMatrix";
    }

private :
    CudaMatrix<T> m;
};

template <class T>
class CudaBaseVector : public BaseVector
{

public :
    CudaVector<T>& getCudaVector()
    {
        return v;
    }

    T& operator[](int i)
    {
        return v[i];
    }

    const T& operator[](int i) const
    {
        return v[i];
    }

    void resize(int nbRow)
    {
        v.resize(nbRow);
    }

    void resize(int nbRow,int warp_size)
    {
        v.resize(nbRow,warp_size);
    }

    unsigned int size() const
    {
        return v.size();
    }

    SReal element(int i) const
    {
        return v[i];
    }

    void clear()
    {
        //for (unsigned int i=0; i<size(); i++) v[i]=(T)(0.0);
        v.memsetHost();
    }

    void set(int i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(int i, SReal val)
    {
        v[i] += (T)val;
    }

    static std::string Name()
    {
        return "CudaBaseVector";
    }

private :
    CudaVector<T> v;
};

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
