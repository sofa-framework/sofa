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
#include "cuda.h"

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

    void ParticleSourceCuda3f_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, float fx, float fy, float fz);
    void ParticleSourceCuda3f_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, float fx, float fy, float fz);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void ParticleSourceCuda3d_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, double fx, double fy, double fz);
    void ParticleSourceCuda3d_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, double fx, double fy, double fz);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////


template<class real>
__global__ void ParticleSourceCuda3t_fillValues_kernel(unsigned int totalsize, unsigned int subsetsize, real* dest, const unsigned int* indices, real fx, real fy, real fz)
{
    unsigned int index0 = umul24(blockIdx.x,BSIZE);
    unsigned int index = index0+threadIdx.x;
    if (index < subsetsize)
    {
        unsigned int dindex = indices[index];
        if (dindex < totalsize)
        {
            unsigned int dindex3 = umul24(dindex,3);
            dest[dindex3+0] = fx;
            dest[dindex3+1] = fy;
            dest[dindex3+2] = fz;
        }
    }
}

template<class real>
__global__ void ParticleSourceCuda3t_copyValuesWithOffset_kernel(unsigned int totalsize, unsigned int subsetsize, real* dest, const unsigned int* indices, const real* src, real fx, real fy, real fz)
{
    unsigned int index0 = umul24(blockIdx.x,BSIZE);
    unsigned int index = index0+threadIdx.x;
    if (index < subsetsize)
    {
        unsigned int dindex = indices[index];
        if (dindex < totalsize)
        {
            unsigned int dindex3 = umul24(dindex,3);
            unsigned int index3 = umul24(index,3);
            dest[dindex3+0] = src[index3+0]+fx;
            dest[dindex3+1] = src[index3+1]+fy;
            dest[dindex3+2] = src[index3+2]+fz;
        }
    }
}

//////////////////////
// CPU-side methods //
//////////////////////


void ParticleSourceCuda3f_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, float fx, float fy, float fz)
{
    dim3 threads(BSIZE,1);
    dim3 grid((subsetsize+BSIZE-1)/BSIZE,1);
    {ParticleSourceCuda3t_fillValues_kernel<float><<< grid, threads >>>(totalsize, subsetsize, (float*)dest, (const unsigned int*)indices, fx, fy, fz); mycudaDebugError("ParticleSourceCuda3t_fillValues_kernel<float>");}
}

void ParticleSourceCuda3f_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, float fx, float fy, float fz)
{
    dim3 threads(BSIZE,1);
    dim3 grid((subsetsize+BSIZE-1)/BSIZE,1);
    {ParticleSourceCuda3t_copyValuesWithOffset_kernel<float><<< grid, threads >>>(totalsize, subsetsize, (float*)dest, (const unsigned int*)indices, (const float*)src, fx, fy, fz); mycudaDebugError("ParticleSourceCuda3t_copyValuesWithOffset_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void ParticleSourceCuda3d_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, double fx, double fy, double fz)
{
    dim3 threads(BSIZE,1);
    dim3 grid((subsetsize+BSIZE-1)/BSIZE,1);
    {ParticleSourceCuda3t_fillValues_kernel<double><<< grid, threads >>>(totalsize, subsetsize, (double*)dest, (const unsigned int*)indices, fx, fy, fz); mycudaDebugError("ParticleSourceCuda3t_fillValues_kernel<double>");}
}

void ParticleSourceCuda3d_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, double fx, double fy, double fz)
{
    dim3 threads(BSIZE,1);
    dim3 grid((subsetsize+BSIZE-1)/BSIZE,1);
    {ParticleSourceCuda3t_copyValuesWithOffset_kernel<double><<< grid, threads >>>(totalsize, subsetsize, (double*)dest, (const unsigned int*)indices, (const double*)src, fx, fy, fz); mycudaDebugError("ParticleSourceCuda3t_copyValuesWithOffset_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
