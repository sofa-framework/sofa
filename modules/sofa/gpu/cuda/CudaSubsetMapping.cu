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
    void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void SubsetMappingCuda3f_apply_kernel(unsigned int size, const int* map, float* out, const float* in)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    float3 res = make_float3(0,0,0);

    int c = map[index0+index1];
    if (index0+index1 < size)
    {
        res = ((const float3*)in)[c];
    }

    //__syncthreads();

    int index3 = umul24(3,index1);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

__global__ void SubsetMappingCuda3f_applyJT_kernel(unsigned int size, unsigned int maxNOut, const int* mapT, float* out, const float* in)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    float3 res = make_float3(0,0,0);
    //res += *((const float3*)in) * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        int data = *mapT;
        mapT+=BSIZE;
        if (data != -1)
            res += ((const float3*)in) [data];
    }

    int index3 = umul24(index1,3);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] += temp[index1        ];
    out[index1+  BSIZE] += temp[index1+  BSIZE];
    out[index1+2*BSIZE] += temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float*)in);
}

void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float*)in);
}

void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_applyJT_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const int*)mapT, (float*)out, (const float*)in);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
