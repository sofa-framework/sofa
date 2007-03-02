#include "CudaCommon.h"
#include "CudaMath.h"

extern "C"
{
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void UniformMassCuda1f_addMDx_kernel(const float mass, float* res, const float* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] += dx[index] * mass;
}

__global__ void UniformMassCuda3f_addMDx_kernel(const float mass, float3* res, const float3* dx)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] += dx[index] * mass;
    float3 dxi = dx[index];
    float3 ri = res[index];
    ri.x += dxi.x * mass;
    ri.y += dxi.y * mass;
    ri.z += dxi.z * mass;
    res[index] = ri;
}

__global__ void UniformMassCuda1f_accFromF_kernel(const float inv_mass, float* a, const float* f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    a[index] = f[index] * inv_mass;
}

__global__ void UniformMassCuda3f_accFromF_kernel(const float inv_mass, float3* a, const float3* f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //a[index] = f[index] * inv_mass;
    float3 fi = f[index];
    float3 ai;
    ai.x = fi.x * inv_mass;
    ai.y = fi.y * inv_mass;
    ai.z = fi.z * inv_mass;
    a[index] = ai;
}

__global__ void UniformMassCuda1f_addForce_kernel(const float mg, float* f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    f[index] += mg;
}

__global__ void UniformMassCuda3f_addForce_kernel(const float3 mg, float* f)
{
    //int index = blockIdx.x*blockDim.x+threadIdx.x;
    //f[index] += mg;
    f += blockIdx.x*BSIZE*3;
    int index = threadIdx.x;
    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];
    temp[index] = f[index];
    temp[index+BSIZE] = f[index+BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    __syncthreads();

    int index3 = 3*index;
    temp[index3+0] += mg.x;
    temp[index3+1] += mg.y;
    temp[index3+2] += mg.z;

    __syncthreads();

    f[index] = temp[index];
    f[index+BSIZE] = temp[index+BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3f_addMDx_kernel<<< grid, threads >>>(mass, (float3*)res, (const float3*)dx);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    UniformMassCuda1f_addMDx_kernel<<< grid, threads >>>(mass, (float*)res, (const float*)dx);
}

void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3f_accFromF_kernel<<< grid, threads >>>(1.0f/mass, (float3*)a, (const float3*)f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    UniformMassCuda1f_accFromF_kernel<<< grid, threads >>>(1.0f/mass, (float*)a, (const float*)f);
}

void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    UniformMassCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(make_float3(mg[0],mg[1],mg[2]), (float*)f);
}
