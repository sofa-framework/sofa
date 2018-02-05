/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

///////////////
//  KERNELS  //
///////////////

extern "C"
{
    void SOFA_GPU_CUDA_API copy_vectorf(int dim,const void * a, void * b);
    void SOFA_GPU_CUDA_API vector_vector_peqf(int dim,float f,const void * a,void * b);
    void SOFA_GPU_CUDA_API sub_vector_vectorf(int dim,const void * a, const void * b, void * r);
    void SOFA_GPU_CUDA_API permute_vectorf(int dim,const void * a, const void * perm, void * b);

#ifdef SOFA_GPU_CUDA_DOUBLE
    void SOFA_GPU_CUDA_API copy_vectord(int dim,const void * a, void * b);
    void SOFA_GPU_CUDA_API vector_vector_peqd(int dim,double f,const void * a,void * b);
    void SOFA_GPU_CUDA_API sub_vector_vectord(int dim,const void * a, const void * b, void * r);
    void SOFA_GPU_CUDA_API permute_vectord(int dim,const void * a, const void * perm, void * b);
#endif
}

template<class real> class CudaVectorUtilsKernels;

template<> class CudaVectorUtilsKernels<float>
{
public:
    // copy the dim first value of a(float) in b(float)
    static void copy_vector(int dim,const void * a,void * b)
    {   copy_vectorf(dim,a,b); }

    // compute b = b + a*f
    static void vector_vector_peq(int dim,float f,const void * a,void * b)
    {   vector_vector_peqf(dim,f,a,b); }

    // compute b = b + a*f
    static void sub_vector_vector(int dim,const void * a,const void * b,void * r)
    {   sub_vector_vectorf(dim,a,b,r); }

    static void permute_vector(int dim,const void * a, const void * perm, void * b)
    {   permute_vectorf(dim,a,perm,b); }
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

    // compute b = b + a*f
    static void sub_vector_vector(int dim,const void * a,const void * b,void * r)
    {   sub_vector_vectord(dim,a,b,r); }

    static void permute_vector(int dim,const void * a, const void * perm, void * b)
    {   permute_vectord(dim,a,perm,b); }
};
#endif

using namespace sofa::defaulttype;

template<class T>
class CudaBaseVectorType : public BaseVector {
public :
    typedef T Real;
    typedef typename BaseVector::Index Index;

    virtual void resize(Index nbRow) = 0;
    virtual Index size() const = 0;
    virtual SReal element(Index i) const = 0;
    virtual void clear() = 0;
    virtual void set(Index i, SReal val) = 0;
    virtual void add(Index i, SReal val) = 0;
    virtual const void* deviceRead(Index off=0) const = 0;
    virtual void * deviceWrite(Index off=0) = 0;
    virtual const T* hostRead(Index off=0) const = 0;
    virtual T * hostWrite(Index off=0) = 0;
    virtual void invalidateDevice() = 0;
    virtual void invalidateHost() = 0;
    virtual T getSingle(Index off=0) = 0;

    /// this += a*f
    template<typename Real>
    void peq(const CudaBaseVectorType<Real> & a, double f) {
        CudaVectorUtilsKernels<Real>::vector_vector_peq(this->size(),
                                                        (Real)f,
                                                        a.deviceRead(),
                                                        this->deviceWrite());
    }

    /// this = a - b
    template<typename Real>
    void sub(const CudaBaseVectorType<Real>& a, const CudaBaseVectorType<Real>& b)
    {
        CudaVectorUtilsKernels<Real>::sub_vector_vector(this->size(),
                                                        a.deviceRead(),
                                                        b.deviceRead(),
                                                        this->deviceWrite());
    }
};

template <class T>
class CudaBaseVector : public CudaBaseVectorType<T>
{
public :
    typedef T Real;
    typedef typename CudaBaseVectorType<T>::Index Index;

    CudaVector<T>& getCudaVector()
    {
        return v;
    }

    const CudaVector<T>& getCudaVector() const
    {
        return v;
    }

    T& operator[](Index i)
    {
        return v[i];
    }

    const T& operator[](Index i) const
    {
        return v[i];
    }

    void fastResize(Index nbRow)
    {
        v.fastResize(nbRow);
    }

    void fastResize(Index nbRow,Index warp_size)
    {
        v.fastResize(nbRow,warp_size);
    }

    void resize(Index nbRow)
    {
        v.resize(nbRow);
    }

    void recreate(Index nbRow)
    {
        v.recreate(nbRow);
    }

    void resize(Index nbRow,Index warp_size)
    {
        v.resize(nbRow,warp_size);
    }

    Index size() const
    {
        return v.size();
    }

    SReal element(Index i) const
    {
        return v[i];
    }

    void clear()
    {
        //for (unsigned Index i=0; i<size(); i++) v[i]=(T)(0.0);
//		  v.memsetHost();
//                    Index size = v.size();
        v.clear();
//                    v.resize(size);
    }

    void set(Index i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(Index i, SReal val)
    {
        v[i] += (T)val;
    }

    void operator=(const CudaBaseVector<Real> & e)
    {
        v = e.v;
    }

    void eq(const CudaBaseVector<Real> & e)
    {
        v = e.v;
    }

    const void* deviceRead(Index off=0) const
    {
        return v.deviceReadAt(off);
    }

    void * deviceWrite(Index off=0)
    {
        return v.deviceWriteAt(off);
    }

    void invalidateDevice()
    {
        v.invalidateDevice();
    }

    void invalidateHost()
    {
        v.invalidateHost();
    }

    void memsetDevice()
    {
        v.memsetDevice();
    }

    const T* hostRead(Index off=0) const
    {
        return v.hostReadAt(off);
    }

    T * hostWrite(Index off=0)
    {
        return v.hostWriteAt(off);
    }

    T getSingle(Index off)
    {
        return v.getSingle(off);
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


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_GPU_CUDA)

extern template class SOFA_GPU_CUDA_API CudaBaseVector< float >;
#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API CudaBaseVector< double >;
#endif

#endif

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
