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

    void LinearMovementConstraintCudaVec6f_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaVec6f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaVec6f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
    void LinearMovementConstraintCudaRigid3f_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaRigid3f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaRigid3f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE
    void LinearMovementConstraintCudaVec6d_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaVec6d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaVec6d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
    void LinearMovementConstraintCudaRigid3d_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaRigid3d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaRigid3d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
#endif // SOFA_GPU_CUDA_DOUBLE

}// extern "C"

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void LinearMovementConstraintCudaVec6t_projectPositionIndexed_kernel(unsigned size, const int* indices, real dirX, real dirY, real dirZ, real dirU, real dirV, real dirW, const CudaVec6<real>* x0, CudaVec6<real>* x)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaVec6<real> m = CudaVec6<real>::make(dirX, dirY, dirZ, dirU, dirV, dirW);
    if (index < size)
    {
        x[indices[index]] = x0[index];
        x[indices[index]] += m;
    }
}// projectPositionIndexed_kernel

template<class real>
__global__ void LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel(unsigned size, const int* indices, CudaRigidDeriv3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    if (index < size)
    {
        dx[indices[index]] = CudaRigidDeriv3<real>::make(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
}// kernel Indexed Cuda3t1

template<class real>
__global__ void LinearMovementConstraintCudaRigid3t_projectPositionIndexed_kernel(unsigned size, const int* indices, real dirX, real dirY, real dirZ, real dirU, real dirV, real dirW, const CudaRigidCoord3<real>* x0, CudaRigidCoord3<real>* x)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaRigidCoord3<real> m = CudaRigidCoord3<real>::make(dirX, dirY, dirZ, dirU, dirV, dirW, 0.0);
    if (index < size)
    {
        x[indices[index]] = x0[index];
        x[indices[index]] += m;
    }
}// projectPositionIndexed_kernel


template<class real>
__global__ void LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel(unsigned size, const int* indices, real velX, real velY, real velZ, real velU, real velV, real velW, CudaRigidDeriv3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaRigidDeriv3<real> vel = CudaRigidDeriv3<real>::make(velX, velY, velZ, velU, velV, velW);
    if (index < size)
    {
        dx[indices[index]] = vel;
    }
}// projectVelocityIndexed_kernel

//////////////////////
// CPU-side methods //
//////////////////////
void LinearMovementConstraintCudaVec6f_projectResponseIndexed(unsigned size, const void* indices, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<float>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<float>");}
}

void LinearMovementConstraintCudaVec6f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    LinearMovementConstraintCudaVec6t_projectPositionIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices,
            ((float*)dir)[0], ((float*)dir)[1], ((float*)dir)[2], ((float*)dir)[3], ((float*)dir)[4], ((float*)dir)[5],
            (const CudaVec6<float>*) x0, (CudaVec6<float>*)x);
}

void LinearMovementConstraintCudaVec6f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, ((float*)dir)[0], ((float*)dir)[1], ((float*)dir)[2], ((float*)dir)[3], ((float*)dir)[4], ((float*)dir)[5], (CudaRigidDeriv3<float>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<float>");}
}
void LinearMovementConstraintCudaRigid3f_projectResponseIndexed(unsigned size, const void* indices, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<float>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<float>");}
}

void LinearMovementConstraintCudaRigid3f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectPositionIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, ((float*)dir)[0], ((float*)dir)[1], ((float*)dir)[2], ((float*)dir)[3], ((float*)dir)[4], ((float*)dir)[5], (const CudaRigidCoord3<float>*) x0, (CudaRigidCoord3<float>*)x); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectPositionIndexed_kernel<float>");}
}

void LinearMovementConstraintCudaRigid3f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, ((float*)dir)[0], ((float*)dir)[1], ((float*)dir)[2], ((float*)dir)[3], ((float*)dir)[4], ((float*)dir)[5], (CudaRigidDeriv3<float>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void LinearMovementConstraintCudaVec6d_projectResponseIndexed(unsigned size, const void* indices, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<double>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<double>");}
}

void LinearMovementConstraintCudaVec6d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    LinearMovementConstraintCudaVec6t_projectPositionIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices,
            ((double*)dir)[0], ((double*)dir)[1], ((double*)dir)[2], ((double*)dir)[3], ((double*)dir)[4], ((double*)dir)[5],
            (const CudaVec6<double>*) x0, (CudaVec6<double>*)x);
}

void LinearMovementConstraintCudaVec6d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, ((double*)dir)[0], ((double*)dir)[1], ((double*)dir)[2], ((double*)dir)[3], ((double*)dir)[4], ((double*)dir)[5], (CudaRigidDeriv3<double>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<double>");}
}

void LinearMovementConstraintCudaRigid3d_projectResponseIndexed(unsigned size, const void* indices, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaRigidDeriv3<double>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectResponseIndexed_kernel<double>");}
}

void LinearMovementConstraintCudaRigid3d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectPositionIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, ((double*)dir)[0], ((double*)dir)[1], ((double*)dir)[2], ((double*)dir)[3], ((double*)dir)[4], ((double*)dir)[5], (const CudaRigidCoord3<double>*) x0, (CudaRigidCoord3<double>*)x); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectPositionIndexed_kernel<double>");}
}

void LinearMovementConstraintCudaRigid3d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, ((double*)dir)[0], ((double*)dir)[1], ((double*)dir)[2], ((double*)dir)[3], ((double*)dir)[4], ((double*)dir)[5], (CudaRigidDeriv3<double>*)dx); mycudaDebugError("LinearMovementConstraintCudaRigid3t_projectVelocityIndexed_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
