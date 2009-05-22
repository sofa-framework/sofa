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
/* PART OF THIS FILE IS FROM NVIDIA CUDA SDK particles demo:
 *
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */


#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"
#include "radixsort.h"

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
    void SpatialGridContainer3f_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x);
    void SpatialGridContainer3f1_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x);
    void SpatialGridContainer3f_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
    void SpatialGridContainer3f1_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
}

#define USE_TEX  1
#define USE_SORT 1

struct GridParams
{
    float cellWidth;
    float invCellWidth;
    int cellMask;
};

//////////////////////
// GPU-side methods //
//////////////////////

#if USE_TEX
#if USE_SORT
texture<uint2, 1, cudaReadModeElementType> particleHashTex;
texture<unsigned int, 1, cudaReadModeElementType> cellStartTex;
#else
texture<unsigned int, 1, cudaReadModeElementType> gridCountersTex;
texture<unsigned int, 1, cudaReadModeElementType> gridCellsTex;
#endif
#endif

__constant__ GridParams gridParams;

// calculate cell in grid from position
__device__ int3 calcGridPos(float3 p)
{
    int3 i;
    i.x = __float2int_rd(p.x * gridParams.invCellWidth);
    i.y = __float2int_rd(p.y * gridParams.invCellWidth);
    i.z = __float2int_rd(p.z * gridParams.invCellWidth);
    return i;
}

// calculate cell in grid from position
__device__ int3 calcGridPos(float4 p)
{
    int3 i;
    i.x = __float2int_rd(p.x * gridParams.invCellWidth);
    i.y = __float2int_rd(p.y * gridParams.invCellWidth);
    i.z = __float2int_rd(p.z * gridParams.invCellWidth);
    return i;
}

// calculate address in grid from position
__device__ unsigned int calcGridHash(int3 p)
{
    //return ((p.x<<10)^(p.y<<5)^(p.z)) & gridParams.cellMask;
    //return ((p.x)^(p.y)^(p.z)) & gridParams.cellMask;
    const unsigned int p0 = 73856093; // large prime numbers
    const unsigned int p1 = 19349663;
    const unsigned int p2 = 83492791;
    return (__mul24(p0,p.x)^__mul24(p1,p.y)^__mul24(p2,p.z)) & gridParams.cellMask;
    //return (p.x) & gridParams.cellMask;
}

// calculate address in grid from position
__device__ unsigned int calcGridHash(float3 p)
{
    return calcGridHash(calcGridPos(p));
}

// calculate address in grid from position
__device__ unsigned int calcGridHash(float4 p)
{
    return calcGridHash(calcGridPos(p));
}

__device__ __inline__ float3 getPos3(const float4* pos, int index0, int index)
{
    float4 p = pos[index];
    return make_float3(p.x,p.y,p.z);
}

__shared__ float ftemp[BSIZE*3];

__device__ __inline__ float3 getPos3(const float3* pos, int index0, int index)
{
    //return pos[index];
    int index03 = __umul24(index0,3);
    int index3 = __umul24(threadIdx.x,3);
    ftemp[threadIdx.x] = ((const float*)pos)[index03+threadIdx.x];
    ftemp[threadIdx.x+BSIZE] = ((const float*)pos)[index03+threadIdx.x+BSIZE];
    ftemp[threadIdx.x+2*BSIZE] = ((const float*)pos)[index03+threadIdx.x+2*BSIZE];
    __syncthreads();
    return make_float3(ftemp[index3],ftemp[index3+1],ftemp[index3+2]);
}

__device__ __inline__ float4 getPos4(const float4* pos, int index0, int index)
{
    return pos[index];
}

__device__ __inline__ float4 getPos4(const float3* pos, int index0, int index)
{
    int index3 = __umul24(threadIdx.x,3);
    pos += index0;
    ftemp[threadIdx.x] = ((const float*)pos)[threadIdx.x];
    ftemp[threadIdx.x+BSIZE] = ((const float*)pos)[threadIdx.x+BSIZE];
    ftemp[threadIdx.x+2*BSIZE] = ((const float*)pos)[threadIdx.x+2*BSIZE];
    __syncthreads();
    return make_float4(ftemp[index3],ftemp[index3+1],ftemp[index3+2],0.0f);
}

__device__ __inline__ float4 getPos4(const float4* pos, int index)
{
    return pos[index];
}

__device__ __inline__ float4 getPos4(const float3* pos, int index)
{
    float3 p = pos[index];
    return make_float4(p.x,p.y,p.z,1.0f);
}

// calculate grid hash value for each particle
template<class TIn>
__global__ void
calcHashD(const TIn* pos,
        uint2*  particleHash, int n)
{
    int index0 = __mul24(blockIdx.x, blockDim.x);
    int index = index0 + threadIdx.x;
    float3 p = getPos3(pos,index0,index);

    // get address in grid
    int3 gridPos = calcGridPos(p);
    unsigned int gridHash = calcGridHash(gridPos);

    // store grid hash and particle index
    particleHash[index] = make_uint2(gridHash, index);
}

// find start of each cell in sorted particle list by comparing with previous hash value
// one thread per particle
__global__ void
findCellStartD(const uint2* particleHash,
        unsigned int * cellStart, int n)
{
    unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i < n)
    {
        volatile uint2 particle = particleHash[i];

        if (i > 0)
        {
            volatile uint2 prevParticle = particleHash[i-1];
            if (particle.x != prevParticle.x)
            {
                cellStart[ particle.x ] = i;
            }
        }
        else
        {
            cellStart[ particle.x ] = i;
        }
    }
}

// rearrange particle data into sorted order
template<class TIn>
__global__ void
reorderDataD(const uint2*  particleHash,  // particle id sorted by hash
        const TIn* oldPos,
        float4* sortedPos, int n
            )
{
    int index0 = __mul24(blockIdx.x, blockDim.x);
    int index = index0 + threadIdx.x;
    if (index < n)
    {
        volatile uint2 sortedData = particleHash[index];
        //float4 pos = getPos4(oldPos,index0,index);
        float4 pos = getPos4(oldPos,sortedData.y);
        sortedPos[index] = pos;
    }
}


//////////////////////
// CPU-side methods //
//////////////////////

void SpatialGridContainer3f_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x)
{
    GridParams p;
    p.cellWidth = cellWidth;
    p.invCellWidth = 1.0f/cellWidth;
    p.cellMask = (1<<cellBits)-1;
    cudaMemcpyToSymbol(gridParams, &p, sizeof(GridParams));
    cudaMemset(cellStart, -1, (1<<cellBits)*sizeof(int));

    // First compute hash of each particle
    {
        dim3 threads(BSIZE,1);
        dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
        calcHashD<float3><<< grid, threads >>>((const float3*)x, (uint2*)particleHash, nbPoints);
    }

    // Then sort it

    //RadixSort((KeyValuePair*)particleHash, (KeyValuePair*)sortTmp, nbPoints, cellBits);

    // Then find the start of each cell
    {
        dim3 threads(BSIZE,1);
        dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
        findCellStartD<<< grid, threads >>>((const uint2*)particleHash, (unsigned int*)cellStart, nbPoints);
    }
}

void SpatialGridContainer3f1_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x)
{
    GridParams p;
    p.cellWidth = cellWidth;
    p.invCellWidth = 1.0f/cellWidth;
    p.cellMask = (1<<cellBits)-1;
    cudaMemcpyToSymbol(gridParams, &p, sizeof(GridParams));
    cudaMemset(cellStart, -1, (1<<cellBits)*sizeof(int));

    // First compute hash of each particle
    {
        dim3 threads(BSIZE,1);
        dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
        calcHashD<float4><<< grid, threads >>>((const float4*)x, (uint2*)particleHash, nbPoints);
    }

    // Then sort it

    //RadixSort((KeyValuePair*)particleHash, (KeyValuePair*)sortTmp, nbPoints, cellBits);

    // Then find the start of each cell
    {
        dim3 threads(BSIZE,1);
        dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
        findCellStartD<<< grid, threads >>>((const uint2*)particleHash, (unsigned int*)cellStart, nbPoints);
    }
}

void SpatialGridContainer3f_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
    reorderDataD<float3><<< grid, threads >>>((const uint2*)particleHash, (const float3*)x, (float4*)sorted, nbPoints);
}

void SpatialGridContainer3f1_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x)
{
    dim3 threads(BSIZE,1);
    dim3 grid((nbPoints+BSIZE-1)/BSIZE,1);
    reorderDataD<float4><<< grid, threads >>>((const uint2*)particleHash, (const float4*)x, (float4*)sorted, nbPoints);
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
