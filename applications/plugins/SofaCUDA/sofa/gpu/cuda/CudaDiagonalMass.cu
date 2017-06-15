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
    void DiagonalMassCuda_addMDxf(unsigned int size, float factor, const void * mass, const void* dx, void* res);
    void DiagonalMassCuda_addMDxd(unsigned int size, double factor, const void * mass, const void* dx, void* res);

    void DiagonalMassCuda_accFromFf(unsigned int size, const void * mass, const void* f, void* a);
    void DiagonalMassCuda_accFromFd(unsigned int size, const void * mass, const void* f, void* a);

    void DiagonalMassCuda_addForcef(unsigned int size, const void * mass,const double * g, const void* f);
    void DiagonalMassCuda_addForced(unsigned int size, const void * mass,const double * g, const void* f);
}


template<class real>
__global__ void DiagonalMassCuda_addMDx_kernel(int size,real factor, const real * mass, const real* dx, real* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index3 = index * 3;
    if (index < size)
    {
        res[index3+0] += dx[index3+0] * mass[index] * factor;
        res[index3+1] += dx[index3+1] * mass[index] * factor;
        res[index3+2] += dx[index3+2] * mass[index] * factor;
    }
}

template<class real>
__global__ void DiagonalMassCuda_accFromF_kernel(int size, const real * inv_mass, const real* f, real* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index3 = index * 3;
    if (index < size)
    {
        a[index3+0] = f[index3+0] / inv_mass[index];
        a[index3+1] = f[index3+1] / inv_mass[index];
        a[index3+2] = f[index3+2] / inv_mass[index];
    }
}

template<class real>
__global__ void DiagonalMassCuda_addForce_kernel(int size, const real * mass, real g_x, real g_y, real g_z, real* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index3 = index * 3;
    if (index < size)
    {
        f[index3+0] += mass[index] * g_x;
        f[index3+1] += mass[index] * g_y;
        f[index3+2] += mass[index] * g_z;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void DiagonalMassCuda_addMDxf(unsigned int size, float factor, const void * mass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_addMDx_kernel<float><<< grid, threads >>>(size,factor, (const float *) mass, (const float*)dx, (float*)res); mycudaDebugError("DiagonalMassCuda_addMDx_kernel<float>");}
}

void DiagonalMassCuda_accFromFf(unsigned int size, const void * mass, const void* f, void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_accFromF_kernel<float><<< grid, threads >>>(size, (const float *) mass, (const float*)f, (float*)a); mycudaDebugError("DiagonalMassCuda_accFromF_kernel<float>");}
}

void DiagonalMassCuda_addForcef(unsigned int size, const void * mass,const double * g, const void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_addForce_kernel<float><<< grid, threads >>>(size,(const float *) mass, g[0],g[1],g[2], (float*)f); mycudaDebugError("DiagonalMassCuda_addForce_kernel<float>");}
}


#ifdef SOFA_GPU_CUDA_DOUBLE

void DiagonalMassCuda_addMDxd(unsigned int size, double factor, const void * mass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_addMDx_kernel<double><<< grid, threads >>>(size,factor, (const double *) mass, (const double*)dx, (double*)res); mycudaDebugError("DiagonalMassCuda_addMDx_kernel<double>");}
}

void DiagonalMassCuda_accFromFd(unsigned int size, const void * mass, const void* f, void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_accFromF_kernel<double><<< grid, threads >>>(size, (const double *) mass, (const double*)f, (double*)a); mycudaDebugError("DiagonalMassCuda_accFromF_kernel<double>");}
}

void DiagonalMassCuda_addForced(unsigned int size, const void * mass,const double * g, const void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {DiagonalMassCuda_addForce_kernel<double><<< grid, threads >>>(size,(const double *) mass, g[0],g[1],g[2], (double*)f); mycudaDebugError("DiagonalMassCuda_addForce_kernel<double>");}
}
#endif

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
