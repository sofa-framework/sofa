/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "cuda.h"
#include "CudaMathRigid.h"

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
    void FixedConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void FixedConstraintCuda1t_projectResponseContiguous_kernel(int size, real* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = 0.0f;
}

template<class real>
__global__ void FixedConstraintCuda3t_projectResponseContiguous_kernel(int size, CudaVec3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda3t1_projectResponseContiguous_kernel(int size, CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCudaRigid3t_projectResponseContiguous_kernel(int size, CudaRigidDeriv3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = CudaRigidDeriv3<real>::make(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda1t_projectResponseIndexed_kernel(int size, const int* indices, real* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = 0.0f;
}

template<class real>
__global__ void FixedConstraintCuda3t_projectResponseIndexed_kernel(int size, const int* indices, CudaVec3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda3t1_projectResponseIndexed_kernel(int size, const int* indices, CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCudaRigid3t_projectResponseIndexed_kernel(int size, const int* indices, CudaRigidDeriv3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaRigidDeriv3<real>::make(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void FixedConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t_projectResponseContiguous_kernel<float><<< grid, threads >>>(size, (CudaVec3<float>*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<float><<< grid, threads >>>(3*size, (float*)dx);
    cudaMemset(dx, 0, size*sizeof(float));
}

void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t_projectResponseContiguous_kernel<float><<< grid, threads >>>(size, (CudaVec3<float>*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<float><<< grid, threads >>>(3*size, (float*)dx);
    cudaMemset(dx, 0, size*3*sizeof(float));
}

void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t1_projectResponseContiguous_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<float><<< grid, threads >>>(4*size, (float*)dx);
    cudaMemset(dx, 0, size*4*sizeof(float));
}

void FixedConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx)
{
//	dim3 threads(BSIZE,1);
    cudaMemset(dx, 0, size*6*sizeof(float));
}

void FixedConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCuda1t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (float*)dx); mycudaDebugError("FixedConstraintCuda1t_projectResponseIndexed_kernel<float>");}
}

void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCuda3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<float>*)dx); mycudaDebugError("FixedConstraintCuda3t_projectResponseIndexed_kernel<float>");}
}

void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCuda3t1_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<float>*)dx); mycudaDebugError("FixedConstraintCuda3t1_projectResponseIndexed_kernel<float>");}
}

void FixedConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCudaRigid3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<float>*)dx); mycudaDebugError("FixedConstraintCudaRigid3t_projectResponseIndexed_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t_projectResponseContiguous_kernel<double><<< grid, threads >>>(size, (CudaVec3<double>*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<double><<< grid, threads >>>(3*size, (double*)dx);
    cudaMemset(dx, 0, size*3*sizeof(double));
}

void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t1_projectResponseContiguous_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<double><<< grid, threads >>>(4*size, (double*)dx);
    cudaMemset(dx, 0, size*4*sizeof(double));
}

void FixedConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx)
{
//	dim3 threads(BSIZE,1);
    cudaMemset(dx, 0, size*6*sizeof(double));
}

void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCuda3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<double>*)dx); mycudaDebugError("FixedConstraintCuda3t_projectResponseIndexed_kernel<double>");}
}

void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCuda3t1_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<double>*)dx); mycudaDebugError("FixedConstraintCuda3t1_projectResponseIndexed_kernel<double>");}
}

void FixedConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {FixedConstraintCudaRigid3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<double>*)dx); mycudaDebugError("FixedConstraintCudaRigid3t_projectResponseIndexed_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
