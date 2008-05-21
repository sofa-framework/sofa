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

    CudaBaseMatrix()
    {
        warp_size = 0;
    }

    CudaMatrix<T> & getCudaMatrix()
    {
        return m;
    }

    void resize(int nbCol, int nbRow)
    {
        m.resize(nbCol,nbRow,warp_size);
    }

    void resize(int nbCol, int nbRow,int ws)
    {
        m.resize(nbCol,nbRow,ws);
    }

    void setwarpsize(int wp)
    {
        warp_size = wp;
    }

    int rowSize() const
    {
        return m.getSizeY();
    }

    int colSize() const
    {
        return m.getSizeX();
    }

    SReal element(int i, int j) const
    {
        return m[i][j];
    }

    void clear()
    {
        for (unsigned j=0; j<m.getSizeX(); j++)
        {
            for (unsigned i=0; i<m.getSizeY(); i++)
            {
                m[j][i] = 0.0;
            }
        }
    }

    void set(int j, int i, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[j][i] = v;
    }

    void add(int j, int i, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[j][i] += v;
    }

private :
    CudaMatrix<T> m;
    int warp_size;
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

    int size() const
    {
        return v.size();
    }

    SReal element(int i) const
    {
        return v[i];
    }

    void clear()
    {
        for (int i=0; i<size(); i++) v[i]=0.0;
    }

    void set(int i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(int i, SReal val)
    {
        v[i] += (T)val;
    }

private :
    CudaVector<T> v;
};

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
