#include "CudaCommon.h"
#include "CudaMath.h"

#if defined(__cplusplus)
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
    void SubsetContactMapperCuda3f_setPoints1(unsigned int size, unsigned int nbTests, unsigned int maxPoints, unsigned int nbPointsPerElem, const void* tests, const void* contacts, void* map);
}

struct __align__(16) GPUContact
{
    int p1;
    float3 p2;
    float distance;
    float3 normal;
};

struct __align__(8) GPUTestEntry
{
    int firstIndex;
    int curSize;
    int maxSize;
    int newIndex;
    int elem1,elem2;
};

//////////////////////
// GPU-side methods //
//////////////////////

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

__global__ void SubsetContactMapperCuda3f_setPoints1_kernel(unsigned int nbPointsPerElem, const GPUTestEntry* tests, const GPUContact* contacts, int* map)
{
    if (threadIdx.x == 0)
        curTestEntry = tests[blockIdx.x];

    __syncthreads();

    GPUContact c = contacts[curTestEntry.firstIndex + threadIdx.x];
    if (threadIdx.x < curTestEntry.curSize)
    {
        map[curTestEntry.newIndex + threadIdx.x] = umul24(curTestEntry.elem1,nbPointsPerElem) + c.p1;
    }
}


//////////////////////
// CPU-side methods //
//////////////////////

void RigidContactMapperCuda3f_setPoints2(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* contacts, void* map)
{
    // round up to 16
    //maxPoints = (maxPoints+15)&-16;
    dim3 threads(maxPoints,1);
    dim3 grid(nbTests,1);
    RigidContactMapperCuda3f_setPoints2_kernel<<< grid, threads >>>((const GPUTestEntry*)tests, (GPUContact*)contacts, (float3*)map);
}

void SubsetContactMapperCuda3f_setPoints1(unsigned int size, unsigned int nbTests, unsigned int maxPoints, unsigned int nbPointsPerElem, const void* tests, const void* contacts, void* map)
{
    // round up to 16
    //maxPoints = (maxPoints+15)&-16;
    dim3 threads(maxPoints,1);
    dim3 grid(nbTests,1);
    SubsetContactMapperCuda3f_setPoints1_kernel<<< grid, threads >>>(nbPointsPerElem, (const GPUTestEntry*)tests, (GPUContact*)contacts, (int*)map);

}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
