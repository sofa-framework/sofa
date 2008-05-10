#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION != 2000
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
    void SubsetMappingCuda3f_applyJT1(unsigned int insize, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f_3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);
}

//////////////////////
// GPU-side methods //
//////////////////////

template<typename TIn>
__global__ void SubsetMappingCuda3f_apply_kernel(unsigned int size, const int* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    float3 res = make_float3(0,0,0);

    int c = map[index0+index1];
    if (index0+index1 < size)
    {
        res = make_float3(in[c]);
    }

    //__syncthreads();

    const int index3 = umul24(index1,3);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

template<typename TIn>
__global__ void SubsetMappingCuda3f1_apply_kernel(unsigned int size, const int* map, float4* out, const TIn* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    float4 res = make_float4(0,0,0,0);

    int c = map[index];
    if (index < size)
    {
        res = make_float4(in[c]);
    }

    out[index] = res;
}

template<typename TIn>
__global__ void SubsetMappingCuda3f_applyJT_kernel(unsigned int size, unsigned int maxNOut, const int* mapT, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

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
            res += make_float3(in[data]);
    }

    const int index3 = umul24(index1,3);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] += temp[index1        ];
    out[index1+  BSIZE] += temp[index1+  BSIZE];
    out[index1+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename TIn>
__global__ void SubsetMappingCuda3f1_applyJT_kernel(unsigned int size, unsigned int maxNOut, const int* mapT, float4* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    float3 res = make_float3(0,0,0);
    //res += *((const float3*)in) * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        int data = *mapT;
        mapT+=BSIZE;
        if (data != -1)
            res += make_float3(in[data]);
    }
    float4 o = out[index];
    o.x += res.x;
    o.y += res.y;
    o.z += res.z;
    out[index] = o;
}

template<typename TOut>
__global__ void SubsetMappingCuda3f_applyJT1_kernel(unsigned int size, const int* map, TOut* out, const float* in)
{

    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    in += umul24(index0,3);
    temp[index1        ] = in[index1        ];
    temp[index1+  BSIZE] = in[index1+  BSIZE];
    temp[index1+2*BSIZE] = in[index1+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(3,index1);
    float3 res = make_float3(temp[index3  ],temp[index3+1],temp[index3+2]);

    int c = map[index0+index1];
    if (index0+index1 < size)
    {
        TOut o = out[c];
        o.x += res.x;
        o.y += res.y;
        o.z += res.z;
        out[c] = o;
    }
}

template<typename TOut>
__global__ void SubsetMappingCuda3f1_applyJT1_kernel(unsigned int size, const int* map, TOut* out, const float4* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    float3 res = make_float3(in[index]);

    int c = map[index];
    if (index < size)
    {
        TOut o = out[c];
        o.x += res.x;
        o.y += res.y;
        o.z += res.z;
        out[c] = o;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float3*)in);
}

void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float3*)in);
}

void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_applyJT_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const int*)mapT, (float*)out, (const float3*)in);
}

void SubsetMappingCuda3f_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_applyJT1_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float3*)out, (const float*)in);
}


void SubsetMappingCuda3f1_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_apply_kernel<float4><<< grid, threads >>>(size, (const int*)map, (float4*)out, (const float4*)in);
}

void SubsetMappingCuda3f1_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_apply_kernel<float4><<< grid, threads >>>(size, (const int*)map, (float4*)out, (const float4*)in);
}

void SubsetMappingCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_applyJT_kernel<float4><<< grid, threads >>>(insize, maxNOut, (const int*)mapT, (float4*)out, (const float4*)in);
}

void SubsetMappingCuda3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_applyJT1_kernel<float4><<< grid, threads >>>(size, (const int*)map, (float4*)out, (const float4*)in);
}


void SubsetMappingCuda3f1_3f_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float4*)in);
}

void SubsetMappingCuda3f1_3f_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_apply_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const float4*)in);
}

void SubsetMappingCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_applyJT_kernel<float3><<< grid, threads >>>(insize, maxNOut, (const int*)mapT, (float4*)out, (const float3*)in);
}

void SubsetMappingCuda3f1_3f_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_applyJT1_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float4*)out, (const float*)in);
}


void SubsetMappingCuda3f_3f1_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_apply_kernel<float3><<< grid, threads >>>(size, (const int*)map, (float4*)out, (const float3*)in);
}

void SubsetMappingCuda3f_3f1_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_apply_kernel<float3><<< grid, threads >>>(size, (const int*)map, (float4*)out, (const float3*)in);
}

void SubsetMappingCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f_applyJT_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const int*)mapT, (float*)out, (const float4*)in);
}

void SubsetMappingCuda3f_3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SubsetMappingCuda3f1_applyJT1_kernel<float3><<< grid, threads >>>(size, (const int*)map, (float3*)out, (const float4*)in);
}

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
