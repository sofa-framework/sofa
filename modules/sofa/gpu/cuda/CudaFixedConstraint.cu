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
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void FixedConstraintCuda1f_projectResponseContiguous_kernel(int size, float* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = 0.0f;
}

__global__ void FixedConstraintCuda3f_projectResponseContiguous_kernel(int size, float3* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = make_float3(0.0f,0.0f,0.0f);
}

__global__ void FixedConstraintCuda3f1_projectResponseContiguous_kernel(int size, float4* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = make_float4(0.0f,0.0f,0.0f,0.0f);
}

__global__ void FixedConstraintCuda1f_projectResponseIndexed_kernel(int size, const int* indices, float* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = 0.0f;
}

__global__ void FixedConstraintCuda3f_projectResponseIndexed_kernel(int size, const int* indices, float3* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = make_float3(0.0f,0.0f,0.0f);
}

__global__ void FixedConstraintCuda3f1_projectResponseIndexed_kernel(int size, const int* indices, float4* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = make_float4(0.0f,0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3f_projectResponseContiguous<<< grid, threads >>>(size, (float3*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1f_projectResponseContiguous_kernel<<< grid, threads >>>(3*size, (float*)dx);
    cudaMemset(dx, 0, size*3*sizeof(float));
}

void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3f1_projectResponseContiguous<<< grid, threads >>>(size, (float4*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1f_projectResponseContiguous_kernel<<< grid, threads >>>(4*size, (float*)dx);
    cudaMemset(dx, 0, size*4*sizeof(float));
}

void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3f_projectResponseIndexed_kernel<<< grid, threads >>>(size, (const int*)indices, (float3*)dx);
}

void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3f1_projectResponseIndexed_kernel<<< grid, threads >>>(size, (const int*)indices, (float4*)dx);
}

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
