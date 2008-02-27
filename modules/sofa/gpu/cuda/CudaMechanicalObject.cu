#include "CudaCommon.h"
#include "CudaMath.h"
#include "mycuda.h"

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
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);

    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f1_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f1_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f1_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f1_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    int MechanicalObjectCudaVec3f1_vDotTmpSize(unsigned int size);
    void MechanicalObjectCudaVec3f1_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* cputmp);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void MechanicalObjectCudaVec1f_vClear_kernel(int size, float* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = 0.0f;
    }
}

__global__ void MechanicalObjectCudaVec3f_vClear_kernel(int size, float3* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = make_float3(0.0f,0.0f,0.0f);
    }
}

__global__ void MechanicalObjectCudaVec3f1_vClear_kernel(int size, float4* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
}

__global__ void MechanicalObjectCudaVec1f_vMEq_kernel(int size, float* res, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
    }
}

__global__ void MechanicalObjectCudaVec3f_vMEq_kernel(int size, float* res, float f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        index += BSIZE;
        res[index] *= f;
        //float3 ri = res[index];
        //ri *= f;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vMEq_kernel(int size, float4* res, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = res[index]*f;
        float4 v = res[index];
        v.x *= f;
        v.y *= f;
        v.z *= f;
        res[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vEqBF_kernel(int size, float* res, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
    }
}

__global__ void MechanicalObjectCudaVec3f_vEqBF_kernel(int size, float* res, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        index += BSIZE;
        res[index] = b[index] * f;
        //float3 bi = b[index];
        //float3 ri = bi * f;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vEqBF_kernel(int size, float4* res, const float4* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = b[index] * f;
        float4 v = b[index];
        v.x *= f;
        v.y *= f;
        v.z *= f;
        res[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vPEq_kernel(int size, float* res, const float* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] += a[index];
    }
}

__global__ void MechanicalObjectCudaVec3f_vPEq_kernel(int size, float* res, const float* a)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        index += BSIZE;
        res[index] += a[index];
        //float3 ai = a[index];
        //float3 ri = res[index];
        //ri += ai;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vPEq_kernel(int size, float4* res, const float4* a)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += a[index];
        float4 v = res[index];
        float4 v2 = a[index];
        v.x += v2.x;
        v.y += v2.y;
        v.z += v2.z;
        res[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vPEqBF_kernel(int size, float* res, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
    }
}

__global__ void MechanicalObjectCudaVec3f_vPEqBF_kernel(int size, float* res, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        index += BSIZE;
        res[index] += b[index] * f;
        //float3 bi = b[index];
        //float3 ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vPEqBF_kernel(int size, float4* res, const float4* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += b[index] * f;
        float4 v = res[index];
        float4 v2 = b[index];
        v.x += v2.x*f;
        v.y += v2.y*f;
        v.z += v2.z*f;
        res[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vAdd_kernel(int size, float* res, const float* a, const float* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
    }
}

__global__ void MechanicalObjectCudaVec3f_vAdd_kernel(int size, float* res, const float* a, const float* b)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        index += BSIZE;
        res[index] = a[index] + b[index];
        //float3 ai = a[index];
        //float3 bi = b[index];
        //float3 ri = ai + bi;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vAdd_kernel(int size, float4* res, const float4* a, const float4* b)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index];
        float4 v = a[index];
        float4 v2 = b[index];
        v.x += v2.x;
        v.y += v2.y;
        v.z += v2.z;
        res[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vOp_kernel(int size, float* res, const float* a, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
    }
}

__global__ void MechanicalObjectCudaVec3f_vOp_kernel(int size, float* res, const float* a, const float* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        index += BSIZE;
        res[index] = a[index] + b[index] * f;
        //float3 ai = a[index];
        //float3 bi = b[index];
        //float3 ri = ai + bi * f;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vOp_kernel(int size, float4* res, const float4* a, const float4* b, float f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        float4 v = a[index];
        float4 v2 = b[index];
        v.x += v2.x*f;
        v.y += v2.y*f;
        v.z += v2.z*f;
        res[index] = v;
    }
}

#define RED_BSIZE 128
#define blockSize RED_BSIZE
//template<unsigned int blockSize>
__global__ void MechanicalObjectCudaVecf_vDot_kernel(unsigned int n, float* res, const float* a, const float* b)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += a[i] * b[i]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32)
    {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) res[blockIdx.x] = sdata[0];
}

//template<unsigned int blockSize>
__global__ void MechanicalObjectCudaVecf_vSum_kernel(int n, float* res, const float* a)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x*(blockSize) + tid;
    unsigned int gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += a[i]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32)
    {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) res[blockIdx.x] = sdata[0];
}

#undef blockSize

//////////////////////
// CPU-side methods //
//////////////////////

void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vAssign_kernel<<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*3*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a)
{
    //dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f1_vAssign_kernel<<< grid, threads >>>(res, a);
    cudaMemcpy(res, a, size*4*sizeof(float), cudaMemcpyDeviceToDevice);
}

void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vClear_kernel<<< grid, threads >>>(size, (float3*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vClear_kernel<<< grid, threads >>>(3*size, (float*)res);
    cudaMemset(res, 0, size*3*sizeof(float));
}

void MechanicalObjectCudaVec3f1_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f1_vClear_kernel<<< grid, threads >>>(size, (float4*)res);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vClear_kernel<<< grid, threads >>>(4*size, (float*)res);
    cudaMemset(res, 0, size*4*sizeof(float));
}

void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vMEq_kernel<<< grid, threads >>>(size, (float*)res, f);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vMEq_kernel<<< grid, threads >>>(3*size, (float*)res, f);
}

void MechanicalObjectCudaVec3f1_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vMEq_kernel<<< grid, threads >>>(size, (float4*)res, f);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vMEq_kernel<<< grid, threads >>>(4*size, (float*)res, f);
}

void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vEqBF_kernel<<< grid, threads >>>(size, (float*)res, (const float*)b, f);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vEqBF_kernel<<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vEqBF_kernel<<< grid, threads >>>(size, (float4*)res, (const float4*)b, f);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vEqBF_kernel<<< grid, threads >>>(4*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vPEq_kernel<<< grid, threads >>>(size, (float*)res, (const float*)a);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vPEq_kernel<<< grid, threads >>>(3*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vPEq_kernel<<< grid, threads >>>(size, (float4*)res, (const float4*)a);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vPEq_kernel<<< grid, threads >>>(4*size, (float*)res, (const float*)a);
}

void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vPEqBF_kernel<<< grid, threads >>>(size, (float*)res, (const float*)b, f);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vPEqBF_kernel<<< grid, threads >>>(3*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vPEqBF_kernel<<< grid, threads >>>(size, (float4*)res, (const float4*)b, f);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vPEqBF_kernel<<< grid, threads >>>(4*size, (float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vAdd_kernel<<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vAdd_kernel<<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec3f1_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vAdd_kernel<<< grid, threads >>>(size, (float4*)res, (const float4*)a, (const float4*)b);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vAdd_kernel<<< grid, threads >>>(4*size, (float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vOp_kernel<<< grid, threads >>>(size, (float*)res, (const float*)a, (const float*)b, f);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vOp_kernel<<< grid, threads >>>(3*size, (float*)res, (const float*)a, (const float*)b, f);
}

void MechanicalObjectCudaVec3f1_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vOp_kernel<<< grid, threads >>>(size, (float4*)res, (const float4*)a, (const float4*)b, f);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vOp_kernel<<< grid, threads >>>(4*size, (float*)res, (const float*)a, (const float*)b, f);
}


int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return (nblocs+2)/3;
}

void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 3;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVecf_vDot_kernel<<< grid, threads, RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            MechanicalObjectCudaVecf_vSum_kernel<<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp);
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

int MechanicalObjectCudaVec3f1_vDotTmpSize(unsigned int size)
{
    size *= 4;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return (nblocs+3)/4;
}

void MechanicalObjectCudaVec3f1_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp, float* rtmp)
{
    size *= 4;
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
        if (nblocs > 256) nblocs = 256;
        dim3 threads(RED_BSIZE,1);
        dim3 grid(nblocs,1);
        //myprintf("size=%d, blocs=%dx%d\n",size,nblocs,RED_BSIZE);
        MechanicalObjectCudaVecf_vDot_kernel<<< grid, threads, RED_BSIZE * sizeof(float) >>>(size, (float*)tmp, (const float*)a, (const float*)b);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            /*
            dim3 threads(RED_BSIZE,1);
            dim3 grid(1,1);
            MechanicalObjectCudaVecf_vSum_kernel<<< grid, threads, RED_BSIZE * sizeof(float) >>>(nblocs, (float*)tmp, (const float*)tmp);
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
            */
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
            //myprintf("dot=%f\n",r);
        }
    }
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
