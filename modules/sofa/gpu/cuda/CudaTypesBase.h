#ifndef SOFA_GPU_CUDA_CUDATYPES_BASE_H
#define SOFA_GPU_CUDA_CUDATYPES_BASE_H

#include "CudaTypes.h"
#include <sofa/component/linearsolver/FullMatrix.h>

namespace sofa
{
namespace gpu
{
namespace cuda
{

using namespace sofa::defaulttype;

class CudaBaseMatrix : public BaseMatrix
{
public :

    CudaBaseMatrix()
    {
        warp_size = 0;
    }

    CudaMatrix<float>& getCudaMatrix()
    {
        return m;
    }

    void resize(int nbRow, int nbCol)
    {
        m.resize(nbCol,nbRow,warp_size);
        //this->clear();
    }

    void setwarpsize(double mu)
    {
        if (mu>0.0) warp_size = 96;
        else warp_size = 64;
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
                m[i][j] = 0.0;
            }
        }
    }

    void set(int i, int j, double v)
    {
        m[i][j] = v;
    }

    void add(int i, int j, double v)
    {
        m[i][j] += v;
    }

private :
    CudaMatrix<float> m;
    int warp_size;
};

class CudaFullVector : public BaseVector
{

public :
    CudaVector<float>& getCudaVector()
    {
        return v;
    }

    float & operator[](int i)
    {
        return v[i];
    }

    const float & operator[](int i) const
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

    void set(int i, double val)
    {
        v[i] = (float) val;
    }

    void add(int i, double val)
    {
        v[i] += (float)val;
    }

private :
    CudaVector<float> v;
};

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
