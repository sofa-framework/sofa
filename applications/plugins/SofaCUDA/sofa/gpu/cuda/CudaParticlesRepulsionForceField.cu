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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

template<class real>
class GPURepulsion
{
public:
    real d;
    real d2;
    real stiffness;
    real damping;
};

typedef GPURepulsion<float> GPURepulsion3f;
typedef GPURepulsion<double> GPURepulsion3d;

extern "C"
{
    void ParticlesRepulsionForceFieldCuda3f_addForce (unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3f* repulsion, void* f, const void* x, const void* v );
    void ParticlesRepulsionForceFieldCuda3f_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3f* repulsion, void* f, const void* x, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void ParticlesRepulsionForceFieldCuda3d_addForce (unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3d* repulsion, void* f, const void* x, const void* v );
    void ParticlesRepulsionForceFieldCuda3d_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3d* repulsion, void* f, const void* x, const void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__device__ void ParticlesRepulsionCalcForce(CudaVec3<real> x1, CudaVec3<real> v1, CudaVec3<real> x2, CudaVec3<real> v2, CudaVec3<real>& force, GPURepulsion<real>& rep)
{
    CudaVec3<real> n = x2-x1;
    real d2 = norm2(n);
    if (d2 < rep.d2)
    {
        CudaVec3<real> vi = v2-v1;
        real inv_d = rsqrtf(d2);
        n *= inv_d;
        real d = d2*inv_d - rep.d;
        real forceIntensity = rep.stiffness*d;
        real dampingIntensity = rep.damping*d;
        force += n*forceIntensity - vi*dampingIntensity;
    }
}

template<class real>
__device__ void ParticlesRepulsionCalcDForce(CudaVec3<real> x1, CudaVec3<real> dx1, CudaVec3<real> x2, CudaVec3<real> dx2, CudaVec3<real>& dforce, GPURepulsion<real>& rep)
{
    CudaVec3<real> n = x2-x1;
    real d2 = norm2(n);
    if (d2 < rep.d2)
    {
        CudaVec3<real> dxi = dx2-dx1;
        //real inv_d = rsqrtf(d2);
        //n *= inv_d;
        //dforce = n * (rep.stiffness * dot(dxi, n));

        //dforce = (n * (rep.stiffness * dot(dxi, n)))/d2;

        dforce += n * (real)__fdividef((rep.stiffness * dot(dxi, n)),d2);
    }
}

template<class real>
__global__ void ParticlesRepulsionForceFieldCuda3t_addForce_kernel(int size, const int *cells, const int *cellGhost, GPURepulsion<real> repulsion, real* f, const real* x, const real* v)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*3];
    __shared__ real temp_v[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
    for (int cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        if (!threadIdx.x)
        {
            //range = *(const int2*)(cells+cell);
            range.x = cells[cell];
            range.y = cells[cell+1];
            range.y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }
        __syncthreads();
        if (range.x <= 0) continue; // no actual particle in this cell
        for (int px0 = range.x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + threadIdx.x;
            CudaVec3<real> xi;
            CudaVec3<real> vi;
            CudaVec3<real> force;
            int index;
            if (px < range.y)
            {
                index = cells[px];
                xi = ((const CudaVec3<real>*)x)[index];
                temp_x[tx3  ] = xi.x;
                temp_x[tx3+1] = xi.y;
                temp_x[tx3+2] = xi.z;
                vi = ((const CudaVec3<real>*)v)[index];
                temp_v[tx3  ] = vi.x;
                temp_v[tx3+1] = vi.y;
                temp_v[tx3+2] = vi.z;
            }
            __syncthreads();
            if (px < ghost)
            {
                // actual particle -> compute interactions
                force = CudaVec3<real>::make(0,0,0);
                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        ParticlesRepulsionCalcForce(xi, vi, ((const CudaVec3<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, repulsion);
                }
            }
            __syncthreads();
            // loop through other groups of particles
            for (int py0 = range.x; py0 < range.y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + threadIdx.x;
                if (py < range.y)
                {
                    int index2 = cells[py];
                    CudaVec3<real> xj = ((const CudaVec3<real>*)x)[index2];
                    temp_x[tx3  ] = xj.x;
                    temp_x[tx3+1] = xj.y;
                    temp_x[tx3+2] = xj.z;
                    CudaVec3<real> vj = ((const CudaVec3<real>*)v)[index2];
                    temp_v[tx3  ] = vj.x;
                    temp_v[tx3+1] = vj.y;
                    temp_v[tx3+2] = vj.z;
                }
                __syncthreads();
                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        ParticlesRepulsionCalcForce(xi, vi, ((const CudaVec3<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_v)[i], force, repulsion);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed force
                ((CudaVec3<real>*)f)[index] += force;
            }
        }
    }
}

template<class real>
__global__ void ParticlesRepulsionForceFieldCuda3t_addDForce_kernel(int size, const int *cells, const int *cellGhost, GPURepulsion<real> repulsion, real* df, const real* x, const real* dx)
{
    __shared__ int2 range;
    __shared__ int ghost;
    __shared__ real temp_x[BSIZE*3];
    __shared__ real temp_dx[BSIZE*3];
    int tx3 = __umul24(threadIdx.x,3);
    for (int cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        if (!threadIdx.x)
        {
            //range = *(const int2*)(cells+cell);
            range.x = cells[cell];
            range.y = cells[cell+1];
            range.y &= ~(1U<<31);
            ghost = cellGhost[cell];
        }
        __syncthreads();
        if (range.x <= 0) continue; // no actual particle in this cell
        for (int px0 = range.x; px0 < ghost; px0 += BSIZE)
        {
            int px = px0 + threadIdx.x;
            CudaVec3<real> xi;
            CudaVec3<real> dxi;
            CudaVec3<real> dforce;
            int index;
            if (px < range.y)
            {
                index = cells[px];
                xi = ((const CudaVec3<real>*)x)[index];
                temp_x[tx3  ] = xi.x;
                temp_x[tx3+1] = xi.y;
                temp_x[tx3+2] = xi.z;
                dxi = ((const CudaVec3<real>*)dx)[index];
                temp_dx[tx3  ] = dxi.x;
                temp_dx[tx3+1] = dxi.y;
                temp_dx[tx3+2] = dxi.z;
            }
            __syncthreads();
            if (px < ghost)
            {
                // actual particle -> compute interactions
                dforce = CudaVec3<real>::make(0,0,0);
                int np = min(range.y-px0,BSIZE);
                for (int i=0; i < np; ++i)
                {
                    if (i != threadIdx.x)
                        ParticlesRepulsionCalcDForce(xi, dxi, ((const CudaVec3<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_dx)[i], dforce, repulsion);
                }
            }
            __syncthreads();
            // loop through other groups of particles
            for (int py0 = range.x; py0 < range.y; py0 += BSIZE)
            {
                if (py0 == px0) continue;
                int py = py0 + threadIdx.x;
                if (py < range.y)
                {
                    int index2 = cells[py];
                    CudaVec3<real> xj = ((const CudaVec3<real>*)x)[index2];
                    temp_x[tx3  ] = xj.x;
                    temp_x[tx3+1] = xj.y;
                    temp_x[tx3+2] = xj.z;
                    CudaVec3<real> dxj = ((const CudaVec3<real>*)dx)[index2];
                    temp_dx[tx3  ] = dxj.x;
                    temp_dx[tx3+1] = dxj.y;
                    temp_dx[tx3+2] = dxj.z;
                }
                __syncthreads();
                if (px < ghost)
                {
                    // actual particle -> compute interactions
                    int np = min(range.y-py0,BSIZE);
                    for (int i=0; i < np; ++i)
                    {
                        ParticlesRepulsionCalcDForce(xi, dxi, ((const CudaVec3<real>*)temp_x)[i], ((const CudaVec3<real>*)temp_dx)[i], dforce, repulsion);
                    }
                }
                __syncthreads();
            }
            if (px < ghost)
            {
                // actual particle -> write computed force
                ((CudaVec3<real>*)df)[index] += dforce;
            }
        }
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void ParticlesRepulsionForceFieldCuda3f_addForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3f* repulsion, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60,1);
    {ParticlesRepulsionForceFieldCuda3t_addForce_kernel<float><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *repulsion, (float*)f, (const float*)x, (const float*)v); mycudaDebugError("ParticlesRepulsionForceFieldCuda3t_addForce_kernel<float>");}
}

void ParticlesRepulsionForceFieldCuda3f_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3f* repulsion, void* df, const void* x, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    {ParticlesRepulsionForceFieldCuda3t_addDForce_kernel<float><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*) cells, (const int*) cellGhost, *repulsion, (float*)df, (const float*)x, (const float*)dx); mycudaDebugError("ParticlesRepulsionForceFieldCuda3t_addDForce_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void ParticlesRepulsionForceFieldCuda3d_addForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3d* repulsion, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    {ParticlesRepulsionForceFieldCuda3t_addForce_kernel<double><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *repulsion, (double*)f, (const double*)x, (const double*)v); mycudaDebugError("ParticlesRepulsionForceFieldCuda3t_addForce_kernel<double>");}
}

void ParticlesRepulsionForceFieldCuda3d_addDForce(unsigned int size, const void* cells, const void* cellGhost, GPURepulsion3d* repulsion, void* df, const void* x, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid(60/BSIZE,1);
    {ParticlesRepulsionForceFieldCuda3t_addDForce_kernel<double><<< grid, threads, BSIZE*3*sizeof(double) >>>(size, (const int*) cells, (const int*) cellGhost, *repulsion, (double*)df, (const double*)x, (const double*)dx); mycudaDebugError("ParticlesRepulsionForceFieldCuda3t_addDForce_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
