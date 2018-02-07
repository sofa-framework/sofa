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
    void MeshMatrixMassCuda_addMDx2f(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res);
    void MeshMatrixMassCuda_addForce2f(int dim, void * f, const void * vertexMass, const double * gravity, float massLumpingCoeff);
    void MeshMatrixMassCuda_accFromF2f(int dim, void * acc, const void * f,  const void * vertexMass, float massLumpingCoeff);
}


template<class real>
__global__ void MeshMatrixMassCuda_addMDx2f_kernel(real factor, real massLumpingCoeff,const real * vertexMass, const real* dx, real* res)
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

template<class real>
__global__ void MeshMatrixMassCuda_addForce2f_kernel(int dim, real *  f, const real * vertexMass, real g_x, real g_y, real massLumpingCoeff)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index2 = index * 2;
    if (index < dim)
    {
        f[index2+0] += vertexMass[index] * massLumpingCoeff * g_x;
        f[index2+1] += vertexMass[index] * massLumpingCoeff * g_y;
    }
}

template<class real>
__global__ void MeshMatrixMassCuda_accFromF2f_kernel(int dim, real * acc, const real * f, const real * vertexMass, real massLumpingCoeff)
{
//    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
//    int index2 = index * 2;
//    if (index < dim) {
//      acc[index2+0] = f[index2+0] / (vertexMass[index2+0] * massLumpingCoeff);
//      acc[index2+1] = f[index2+1] / (vertexMass[index2+1] * massLumpingCoeff);
//    }
    int tx = threadIdx.x;
    int tx2 = tx>>1;
    int index1 = umul24(blockIdx.x,BSIZE);
    int index2 = index1<<1;

    __shared__ real s_f[BSIZE];
    __shared__ real s_vertexMass[BSIZE];

    s_vertexMass[tx] = vertexMass[index1+tx];
    s_f[tx] = f[index2+tx];
    __syncthreads();

    //LUMPING INTEGRATION METHOD-------------------------------
    acc[index2+tx] = s_f[tx] / (s_vertexMass[tx2] * massLumpingCoeff);

    //__syncthreads();

    index2 += BSIZE;
    tx2 += BSIZE>>1;
    s_f[tx] = f[index2+tx];

    //__syncthreads();

    acc[index2+tx] = s_f[tx] / (s_vertexMass[tx2] * massLumpingCoeff);
}



//////////////////////
// CPU-side methods //
//////////////////////

void MeshMatrixMassCuda_addMDx2f(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMatrixMassCuda_addMDx2f_kernel<float><<< grid, threads >>>(factor, massLumpingCoeff, (const float *) vertexMass, (const float *) dx, (float*) res);    mycudaDebugError("MeshMatrixMassCuda_addMDx2f_kernel<float>");}
}

void MeshMatrixMassCuda_addForce2f(int dim, void * f, const void * vertexMass, const double * g, float massLumpingCoeff)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);
    {MeshMatrixMassCuda_addForce2f_kernel<float><<< grid, threads >>>(dim, (float *) f, (const float *) vertexMass, g[0], g[1], massLumpingCoeff);              mycudaDebugError("MeshMatrixMassCuda_addForce2f_kernel<float>");}
}

void MeshMatrixMassCuda_accFromF2f(int dim, void * acc, const void * f,  const void * vertexMass, float massLumpingCoeff)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);
    {MeshMatrixMassCuda_accFromF2f_kernel<float><<< grid, threads >>>(dim, (float *) acc, (const float *) f, (const float *) vertexMass, massLumpingCoeff);     mycudaDebugError("MeshMatrixMassCuda_accFromF2f_kernel<float>");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
