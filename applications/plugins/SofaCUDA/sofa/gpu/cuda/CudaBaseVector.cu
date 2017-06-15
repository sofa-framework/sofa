/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "CudaCommon.h"
#include "CudaMath.h"
#include "mycuda.h"
#include <cuda.h>

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

using namespace sofa::gpu::cuda;

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


template<class real>
__global__ void Cuda_CopyVector_kernel(int dim, const real * a, real * b)
{
    int ti = umul24(blockIdx.x,BSIZE) + threadIdx.x;
    if (ti >= dim) return;
    b[ti] = a[ti];
}

void SOFA_GPU_CUDA_API copy_vectorf(int dim,const void * a, void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_CopyVector_kernel<float><<< grid, threads >>>(dim,(const float *) a,(float *) b); mycudaDebugError("Cuda_CopyVector_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void SOFA_GPU_CUDA_API copy_vectord(int dim,const void * a, void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_CopyVector_kernel<double><<< grid, threads >>>(dim,(const double *) a,(double *) b); mycudaDebugError("Cuda_CopyVector_kernel<double>");}
}
#endif

template<class real>
__global__ void Cuda_vector_vector_peq_kernel(int dim,real f, const real * a, real * b)
{
    int ti = umul24(blockIdx.x,BSIZE) + threadIdx.x;
    if (ti >= dim) return;
    b[ti] += a[ti]*f;
}

void vector_vector_peqf(int dim,float f,const void * a,void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_vector_vector_peq_kernel<float><<< grid, threads >>>(dim,f,(const float *) a,(float *) b); mycudaDebugError("Cuda_vector_vector_peq_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void SOFA_GPU_CUDA_API vector_vector_peqd(int dim,double f,const void * a,void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_vector_vector_peq_kernel<double><<< grid, threads >>>(dim,f,(const double *) a,(double *) b); mycudaDebugError("Cuda_vector_vector_peq_kernel<double>");}
}
#endif


template<class real>
__global__ void Cuda_sub_vector_kernel(int dim,const real * a, const real * b, real * r)
{
    int ti = umul24(blockIdx.x,BSIZE) + threadIdx.x;
    if (ti >= dim) return;
    r[ti] = a[ti] - b[ti];
}

void sub_vector_vectorf(int dim,const void * a, const void * b, void * r)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_sub_vector_kernel<float><<< grid, threads >>>(dim,(const float *) a,(const float *) b,(float *) r); mycudaDebugError("Cuda_sub_vector_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void SOFA_GPU_CUDA_API sub_vector_vectord(int dim,const void * a, const void * b, void * r)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {Cuda_sub_vector_kernel<double><<< grid, threads >>>(dim,(const double *) a,(const double *) b,(double *) r); mycudaDebugError("Cuda_sub_vector_kernel<double>");}
}
#endif


template<class real>
__global__ void permute_vector_kernel(int dim,const real * a, const int * perm, real * b)
{
    int ti = umul24(blockIdx.x,BSIZE) + threadIdx.x;
    if (ti >= dim) return;
    b[ti] = a[perm[ti]];
}

void permute_vectorf(int dim,const void * a, const void * perm, void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {permute_vector_kernel<float><<< grid, threads >>>(dim,(const float *) a,(const int *) perm, (float *) b); mycudaDebugError("Cuda_sub_vector_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void SOFA_GPU_CUDA_API permute_vectord(int dim,const void * a, const void * perm, void * b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    {permute_vector_kernel<double><<< grid, threads >>>(dim,(const double *) a,(const int *) perm,(double *) b); mycudaDebugError("Cuda_sub_vector_kernel<double>");}
}
#endif

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
