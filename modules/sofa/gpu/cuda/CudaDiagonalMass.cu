/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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


}


template<class real>
__global__ void DiagonalMassCuda_addMDx_kernel(int size,real factor, const real * mass, const real* dx, real* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size) res[index] += dx[index] * mass[index] * factor;
}

template<class real>
__global__ void DiagonalMassCuda_accFromF_kernel(int size, const real * inv_mass, const real* f, real* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size) a[index] = f[index] / inv_mass[index];
}

// template<class real>
// __global__ void DiagonalMassCuda_addForce_kernel(int size, const real mg, real* f) {
// 	int index = umul24(blockIdx.x,BSIZE);
// 	if (index < size) f[index] += mg;
// }

//////////////////////
// CPU-side methods //
//////////////////////

void DiagonalMassCuda_addMDxf(unsigned int size, float factor, const void * mass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    DiagonalMassCuda_addMDx_kernel<float><<< grid, threads >>>(size,factor, (const float *) mass, (const float*)dx, (float*)res);
}

void DiagonalMassCuda_accFromFf(unsigned int size, const void * mass, const void* f, void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    DiagonalMassCuda_accFromF_kernel<float><<< grid, threads >>>(size, (const float *) mass, (const float*)f, (float*)a);
}

// void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f) {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	UniformMassCuda3t_addForce_kernel<float><<< grid, threads >>>(size, CudaVec3<float>::make(mg[0],mg[1],mg[2]), (float*)f);
// }


#ifdef SOFA_GPU_CUDA_DOUBLE

void DiagonalMassCuda_addMDxd(unsigned int size, double factor, const void * mass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    DiagonalMassCuda_addMDx_kernel<double><<< grid, threads >>>(size,factor, (const double *) mass, (const double*)dx, (double*)res);
}

void DiagonalMassCuda_accFromFd(unsigned int size, const void * mass, const void* f, void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    DiagonalMassCuda_accFromF_kernel<double><<< grid, threads >>>(size, (const double *) mass, (const double*)f, (double*)a);
}
#endif

// void UniformMassCuda3d_addForced(unsigned int size, const double *mg, void* f) {
// 	dim3 threads(BSIZE,1);
// 	dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	UniformMassCuda3t_addForce_kernel<double><<< grid, threads >>>(size, CudaVec3<double>::make(mg[0],mg[1],mg[2]), (double*)f);
// }

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
