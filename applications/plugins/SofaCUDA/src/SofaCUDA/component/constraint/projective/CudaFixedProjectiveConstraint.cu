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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include "cuda.h"
#include <sofa/gpu/cuda/CudaMathRigid.h>

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void FixedProjectiveConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedProjectiveConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void FixedProjectiveConstraintCuda1t_projectResponseContiguous_kernel(int size, real* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[index] = 0.0f;
}

template<class real>
__global__ void FixedProjectiveConstraintCuda3t_projectResponseContiguous_kernel(int size, CudaVec3<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedProjectiveConstraintCuda3t1_projectResponseContiguous_kernel(int size, CudaVec4<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedProjectiveConstraintCudaRigid3t_projectResponseContiguous_kernel(int size, CudaRigidDeriv3<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[index] = CudaRigidDeriv3<real>::make(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedProjectiveConstraintCuda1t_projectResponseIndexed_kernel(int size, const int* indices, real* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[indices[index]] = 0.0f;
}

template<class real>
__global__ void FixedProjectiveConstraintCuda3t_projectResponseIndexed_kernel(int size, const int* indices, CudaVec3<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedProjectiveConstraintCuda3t1_projectResponseIndexed_kernel(int size, const int* indices, CudaVec4<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedProjectiveConstraintCudaRigid3t_projectResponseIndexed_kernel(int size, const int* indices, CudaRigidDeriv3<real>* dx)
{
    int index = blockIdx.x * BSIZE+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaRigidDeriv3<real>::make(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void FixedProjectiveConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*sizeof(float));
}

void FixedProjectiveConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*3*sizeof(float));
}

void FixedProjectiveConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*4*sizeof(float));
}

void FixedProjectiveConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*6*sizeof(float));
}

void FixedProjectiveConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCuda1t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (float*)dx); mycudaDebugError("FixedProjectiveConstraintCuda1t_projectResponseIndexed_kernel<float>");}
}

void FixedProjectiveConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCuda3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<float>*)dx); mycudaDebugError("FixedProjectiveConstraintCuda3t_projectResponseIndexed_kernel<float>");}
}

void FixedProjectiveConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCuda3t1_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<float>*)dx); mycudaDebugError("FixedProjectiveConstraintCuda3t1_projectResponseIndexed_kernel<float>");}
}

void FixedProjectiveConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCudaRigid3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<float>*)dx); mycudaDebugError("FixedProjectiveConstraintCudaRigid3t_projectResponseIndexed_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void FixedProjectiveConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*3*sizeof(double));
}

void FixedProjectiveConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*4*sizeof(double));
}

void FixedProjectiveConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx)
{
    cudaMemset(dx, 0, size*6*sizeof(double));
}

void FixedProjectiveConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCuda3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<double>*)dx); mycudaDebugError("FixedProjectiveConstraintCuda3t_projectResponseIndexed_kernel<double>");}
}

void FixedProjectiveConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCuda3t1_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<double>*)dx); mycudaDebugError("FixedProjectiveConstraintCuda3t1_projectResponseIndexed_kernel<double>");}
}

void FixedProjectiveConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedProjectiveConstraintCudaRigid3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<double>*)dx); mycudaDebugError("FixedProjectiveConstraintCudaRigid3t_projectResponseIndexed_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
