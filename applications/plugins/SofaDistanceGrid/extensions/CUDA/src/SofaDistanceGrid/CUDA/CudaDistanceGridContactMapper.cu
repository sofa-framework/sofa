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
#include <vector_types.h>
#include <sofa/gpu/cuda/mycuda.h>
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
    void RigidContactMapperCuda3f_setPoints2(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* contacts, void* map);
}

struct /*__align__(16)*/ GPUContact
{
    int p1;
    float3 p2;
    float distance;
    float3 normal;
};

struct /*__align__(8)*/ GPUTestEntry
{
    int firstIndex;
    int curSize;
    int maxSize;
    int newIndex;
    int elem1,elem2;
};

__shared__ GPUTestEntry curTestEntry;

__global__ void RigidContactMapperCuda3f_setPoints2_kernel(const GPUTestEntry* tests, const GPUContact* contacts, float3* map)
{
    if (threadIdx.x == 0)
        curTestEntry = tests[blockIdx.x];

    __syncthreads();

    GPUContact c = contacts[curTestEntry.firstIndex + threadIdx.x];
    if (threadIdx.x < curTestEntry.curSize)
    {
        map[curTestEntry.newIndex + threadIdx.x] = c.p2;
    }
}

void RigidContactMapperCuda3f_setPoints2(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* contacts, void* map)
{
    // round up to 16
    //maxPoints = (maxPoints+15)&-16;
    dim3 threads(maxPoints,1);
    dim3 grid(nbTests,1);
    {RigidContactMapperCuda3f_setPoints2_kernel<<< grid, threads >>>((const GPUTestEntry*)tests, (GPUContact*)contacts, (float3*)map); mycudaDebugError("RigidContactMapperCuda3f_setPoints2_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
}
}
}
#endif
