#include "CudaCommon.h"
#include "CudaMath.h"

extern "C"
{
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void FixedConstraintCuda1f_projectResponseContiguous_kernel(float* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    dx[index] = 0.0f;
}

__global__ void FixedConstraintCuda3f_projectResponseContiguous_kernel(float3* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    dx[index] = make_float3(0.0f,0.0f,0.0f);
}

__global__ void FixedConstraintCuda1f_projectResponseIndexed_kernel(const int* indices, float* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    dx[indices[index]] = 0.0f;
}

__global__ void FixedConstraintCuda3f_projectResponseIndexed_kernel(const int* indices, float3* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    dx[indices[index]] = make_float3(0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3f_projectResponseContiguous<<< grid, threads >>>((float3*)dx);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda1f_projectResponseContiguous_kernel<<< grid, threads >>>((float*)dx);
}

void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3f_projectResponseIndexed_kernel<<< grid, threads >>>((const int*)indices, (float3*)dx);
}
