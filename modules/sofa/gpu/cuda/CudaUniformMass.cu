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
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);

    void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void UniformMassCuda1f_addMDx_kernel(int size, const float mass, float* res, const float* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        res[index] += dx[index] * mass;
    }
}

__global__ void UniformMassCuda3f_addMDx_kernel(int size, const float mass, float3* res, const float3* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //res[index] += dx[index] * mass;
        float3 dxi = dx[index];
        float3 ri = res[index];
        ri += dxi * mass;
        res[index] = ri;
    }
}

__global__ void UniformMassCuda3f1_addMDx_kernel(int size, const float mass, float4* res, const float4* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //res[index] += dx[index] * mass;
        float4 dxi = dx[index];
        float4 ri = res[index];
        ri.x += dxi.x * mass;
        ri.y += dxi.y * mass;
        ri.z += dxi.z * mass;
        res[index] = ri;
    }
}

__global__ void UniformMassCuda1f_accFromF_kernel(int size, const float inv_mass, float* a, const float* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        a[index] = f[index] * inv_mass;
    }
}

__global__ void UniformMassCuda3f_accFromF_kernel(int size, const float inv_mass, float3* a, const float3* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //a[index] = f[index] * inv_mass;
        float3 fi = f[index];
        fi *= inv_mass;
        a[index] = fi;
    }
}

__global__ void UniformMassCuda3f1_accFromF_kernel(int size, const float inv_mass, float4* a, const float4* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //a[index] = f[index] * inv_mass;
        float4 fi = f[index];
        fi.x *= inv_mass;
        fi.y *= inv_mass;
        fi.z *= inv_mass;
        a[index] = fi;
    }
}

__global__ void UniformMassCuda1f_addForce_kernel(int size, const float mg, float* f)
{
    int index = umul24(blockIdx.x,BSIZE);
    if (index < size)
    {
        f[index] += mg;
    }
}

__global__ void UniformMassCuda3f_addForce_kernel(int size, const float3 mg, float* f)
{
    //int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //f[index] += mg;
    f += umul24(blockIdx.x,BSIZE*3); //blockIdx.x*BSIZE*3;
    int index = threadIdx.x;
    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];
    temp[index] = f[index];
    temp[index+BSIZE] = f[index+BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    __syncthreads();

    if (umul24(blockIdx.x,BSIZE)+threadIdx.x < size)
    {
        int index3 = umul24(index,3); //3*index;
        temp[index3+0] += mg.x;
        temp[index3+1] += mg.y;
        temp[index3+2] += mg.z;
    }

    __syncthreads();

    f[index] = temp[index];
    f[index+BSIZE] = temp[index+BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}

__global__ void UniformMassCuda3f1_addForce_kernel(int size, const float3 mg, float4* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //f[index] += mg;
        float4 fi = f[index];
        fi.x += mg.x;
        fi.y += mg.y;
        fi.z += mg.z;
        f[index] = fi;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3f_addMDx_kernel<<< grid, threads >>>(size, mass, (float3*)res, (const float3*)dx);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    UniformMassCuda1f_addMDx_kernel<<< grid, threads >>>(3*size, mass, (float*)res, (const float*)dx);
}

void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    UniformMassCuda3f1_addMDx_kernel<<< grid, threads >>>(size, mass, (float4*)res, (const float4*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1f_addMDx_kernel<<< grid, threads >>>(4*size, mass, (float*)res, (const float*)dx);
}

void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3f_accFromF_kernel<<< grid, threads >>>(size, 1.0f/mass, (float3*)a, (const float3*)f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    UniformMassCuda1f_accFromF_kernel<<< grid, threads >>>(3*size, 1.0f/mass, (float*)a, (const float*)f);
}

void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    UniformMassCuda3f1_accFromF_kernel<<< grid, threads >>>(size, 1.0f/mass, (float4*)a, (const float4*)f);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1f_accFromF_kernel<<< grid, threads >>>(4*size, 1.0f/mass, (float*)a, (const float*)f);
}

void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    UniformMassCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, make_float3(mg[0],mg[1],mg[2]), (float*)f);
}

void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    UniformMassCuda3f1_addForce_kernel<<< grid, threads >>>(size, make_float3(mg[0],mg[1],mg[2]), (float4*)f);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
