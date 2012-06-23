/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_GPU_CUDA_CUDABASEVECTOR_H
#define SOFA_GPU_CUDA_CUDABASEVECTOR_H

#include "CudaTypes.h"
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class real> class CudaVectorUtilsKernels;

using namespace sofa::defaulttype;

template <class T>
class CudaBaseVector : public BaseVector
{
public :
    typedef T Real;

    CudaVector<T>& getCudaVector()
    {
        return v;
    }

    const CudaVector<T>& getCudaVector() const
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

    void fastResize(int nbRow)
    {
        v.fastResize(nbRow);
    }

    void fastResize(int nbRow,int warp_size)
    {
        v.fastResize(nbRow,warp_size);
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
//		  v.memsetHost();
        v.clear();
    }

    void set(int i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(int i, SReal val)
    {
        v[i] += (T)val;
    }

    /// v += a*f
    template<typename Real2,typename Real3>
    void peq(const CudaBaseVector<Real2>& a, Real3 f)
    {
        CudaVectorUtilsKernels<Real>::vector_vector_peq(v.size(),
                (Real)f,
                a.deviceRead(),
                this->deviceWrite());
    }

    void operator=(const CudaBaseVector<Real> & e)
    {
        v = e.v;
    }

    const void* deviceRead(int off=0) const
    {
        return v.deviceReadAt(off);
    }

    void * deviceWrite(int off=0)
    {
        return v.deviceWriteAt(off);
    }

    void invalidateDevice()
    {
        v.invalidateDevice();
    }

    const T* hostRead(int off=0) const
    {
        return v.hostReadAt(off);
    }

    T * hostWrite(int off=0)
    {
        return v.hostWriteAt(off);
    }

    static const char* Name(); /* {
			return "CudaBaseVector";
            }*/

    friend std::ostream& operator<< ( std::ostream& os, const CudaBaseVector<T> & vec )
    {
        os << vec.v;
        return os;
    }

private :
    CudaVector<T> v;
};

typedef CudaBaseVector<float> CudaBaseVectorf;
#ifdef SOFA_GPU_CUDA_DOUBLE
typedef CudaBaseVector<double> CudaBaseVectord;
#endif

template<> inline const char* CudaBaseVectorf::Name() { return "CudaBaseVectorf"; }
#ifdef SOFA_GPU_CUDA_DOUBLE
template<> inline const char* CudaBaseVectord::Name() { return "CudaBaseVectord"; }
#endif


///////////////
//  KERNELS  //
///////////////

extern "C"
{
    void copy_vectorf(int dim,const void * a, void * b);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void copy_vectord(int dim,const void * a, void * b);
#endif

    void vector_vector_peqf(int dim,float f,const void * a,void * b);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void vector_vector_peqd(int dim,double f,const void * a,void * b);
#endif
}


template<> class CudaVectorUtilsKernels<float>
{
public:
    // copy the dim first value of a(float) in b(float)
    static void copy_vector(int dim,const void * a,void * b)
    {   copy_vectorf(dim,a,b); }

    // compute b = b + a*f
    static void vector_vector_peq(int dim,float f,const void * a,void * b)
    {   vector_vector_peqf(dim,f,a,b); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE
template<> class CudaVectorUtilsKernels<double>
{
public:
    // copy the dim first value of a(float) in b(float)
    static void copy_vector(int dim,const void * a,void * b)
    {   copy_vectord(dim,a,b); }

    // compute b = b + a*f
    static void vector_vector_peq(int dim,double f,const void * a,void * b)
    {   vector_vector_peqd(dim,f,a,b); }
};
#endif

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
