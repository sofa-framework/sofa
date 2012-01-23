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
#include "CudaCommon.h"
#include "CudaMath.h"
#include "mycuda.h"
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
    void MeshMatrixMassCuda_addMDxf(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res);
}


template<class real>
__global__ void MeshMatrixMassCuda_addMDx_kernel(int size, real factor, real massLumpingCoeff,const real * vertexMass, const real* dx, real* res)
{
    int tx = threadIdx.x;
    int tx2 = tx>>1;
    int index1 = umul24(blockIdx.x,BSIZE);
    int index2 = index1<<1;

    __shared__ real s_dx[BSIZE];
    __shared__ real s_vertexMass[BSIZE];

    s_vertexMass[tx] = vertexMass[index1+tx];
    s_dx[tx] = dx[index2+tx];
    __syncthreads();

    //LUMPING INTEGRATION METHOD-------------------------------
    res[index2+tx] += s_dx[tx] * s_vertexMass[tx2] * massLumpingCoeff * factor;

    //__syncthreads();

    index2 += BSIZE;
    tx2 += BSIZE>>1;
    s_dx[tx] = dx[index2+tx];

    //__syncthreads();

    res[index2+tx] += s_dx[tx] * s_vertexMass[tx2] * massLumpingCoeff * factor;
}



//////////////////////
// CPU-side methods //
//////////////////////

void MeshMatrixMassCuda_addMDxf(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMatrixMassCuda_addMDx_kernel<float><<< grid, threads >>>(size, factor, massLumpingCoeff, (const float *) vertexMass, (const float *) dx, (float*) res);}
}



#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
