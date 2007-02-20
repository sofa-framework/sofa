#define BSIZE 16
#define MAXTHREADS 512

//#include <alloca.h>
#include <malloc.h>

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
    void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void MechanicalObjectCudaVec1f_vClear_kernel(float* res)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] = 0.0f;
}

__global__ void MechanicalObjectCudaVec3f_vClear_kernel(float3* res)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] = make_float3(0.0f,0.0f,0.0f);
}

__global__ void MechanicalObjectCudaVec1f_vMEq_kernel(float* res, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] *= f;
}

__global__ void MechanicalObjectCudaVec3f_vMEq_kernel(float3* res, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] *= f;
    float3 ri = res[index];
    ri.x *= f;
    ri.y *= f;
    ri.z *= f;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vEqBF_kernel(float* res, const float* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] = b[index] * f;
}

__global__ void MechanicalObjectCudaVec3f_vEqBF_kernel(float3* res, const float3* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] = b[index] * f;
    float3 bi = b[index];
    float3 ri;
    ri.x = bi.x * f;
    ri.y = bi.y * f;
    ri.z = bi.z * f;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vPEq_kernel(float* res, const float* a)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] += a[index];
}

__global__ void MechanicalObjectCudaVec3f_vPEq_kernel(float3* res, const float3* a)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] += a[index];
    float3 ai = a[index];
    float3 ri = res[index];
    ri.x += ai.x;
    ri.y += ai.y;
    ri.z += ai.z;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vPEqBF_kernel(float* res, const float* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] += b[index] * f;
}

__global__ void MechanicalObjectCudaVec3f_vPEqBF_kernel(float3* res, const float3* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] += b[index] * f;
    float3 bi = b[index];
    float3 ri = res[index];
    ri.x += bi.x * f;
    ri.y += bi.y * f;
    ri.z += bi.z * f;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vAdd_kernel(float* res, const float* a, const float* b)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] = a[index] + b[index];
}

__global__ void MechanicalObjectCudaVec3f_vAdd_kernel(float3* res, const float3* a, const float3* b)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] = a[index] + b[index];
    float3 ai = a[index];
    float3 bi = b[index];
    float3 ri;
    ri.x = ai.x + bi.x;
    ri.y = ai.y + bi.y;
    ri.z = ai.z + bi.z;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vOp_kernel(float* res, const float* a, const float* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    res[index] = a[index] + b[index] * f;
}

__global__ void MechanicalObjectCudaVec3f_vOp_kernel(float3* res, const float3* a, const float3* b, float f)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    //res[index] = a[index] + b[index] * f;
    float3 ai = a[index];
    float3 bi = b[index];
    float3 ri;
    ri.x = ai.x + bi.x * f;
    ri.y = ai.y + bi.y * f;
    ri.z = ai.z + bi.z * f;
    res[index] = ri;
}

__global__ void MechanicalObjectCudaVec1f_vDot_kernel(float* res, const float* a, const float* b, int size, int offset)
{
    //! Dynamically allocated shared memory for gather
    extern  __shared__  float temp[];
    int index0 = blockIdx.x*blockDim.x;
    int index1 = threadIdx.x;
    int n = min(blockDim.x , size-index0);

    int index = index0+index1;
    float acc = a[index] * b[index];

    while(offset>0)
    {
        if (index1 >= offset && index1 < n)
            temp[index1] = acc;
        __syncthreads();
        if (index1+offset < n)
            acc += temp[index1+offset];
        n = offset;
        offset >>= 1;
    }
    if (index1 == 0)
        res[index0] = acc;
}

__global__ void MechanicalObjectCudaVec3f_vDot_kernel(float* res, const float* a, const float* b, int size, int offset)
{
    //! Dynamically allocated shared memory for gather
    extern  __shared__  float temp[];
    int index0 = blockIdx.x*blockDim.x;
    int index1 = threadIdx.x;
    int n = min(blockDim.x , size-index0);

    int index = index0*3+index1;
    float acc = a[index] * b[index];
    index += n;
    acc += a[index] * b[index];
    index += n;
    acc += a[index] * b[index];

    while(offset>0)
    {
        if (index1 >= offset && index1 < n)
            temp[index1] = acc;
        __syncthreads();
        if (index1+offset < n)
            acc += temp[index1+offset];
        n = offset;
        offset >>= 1;
    }
    if (index1 == 0)
        res[index0] = acc;
}

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

void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vClear_kernel<<< grid, threads >>>((float3*)res);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vClear_kernel<<< grid, threads >>>((float*)res);
}

void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vMEq_kernel<<< grid, threads >>>((float3*)res, f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vMEq_kernel<<< grid, threads >>>((float*)res, f);
}

void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vEqBF_kernel<<< grid, threads >>>((float3*)res, (const float3*)b, f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vEqBF_kernel<<< grid, threads >>>((float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vPEq_kernel<<< grid, threads >>>((float3*)res, (const float3*)a);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vPEq_kernel<<< grid, threads >>>((float*)res, (const float*)a);
}

void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vPEqBF_kernel<<< grid, threads >>>((float3*)res, (const float3*)b, f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vPEqBF_kernel<<< grid, threads >>>((float*)res, (const float*)b, f);
}

void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vAdd_kernel<<< grid, threads >>>((float3*)res, (const float3*)a, (const float3*)b);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vAdd_kernel<<< grid, threads >>>((float*)res, (const float*)a, (const float*)b);
}

void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec3f_vOp_kernel<<< grid, threads >>>((float3*)res, (const float3*)a, (const float3*)b, f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec1f_vOp_kernel<<< grid, threads >>>((float*)res, (const float*)a, (const float*)b, f);
}

int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size)
{
    int nblocs = (size+MAXTHREADS-1)/MAXTHREADS;
    return (nblocs+2)/3;
}

void MechanicalObjectCudaVec3f_vDot(unsigned int size, float* res, const void* a, const void* b, void* tmp)
{
    if (size==0)
    {
        *res = 0.0f;
    }
    else
    {
        int nblocs = (size+MAXTHREADS-1)/MAXTHREADS;
        int bsize = (size+nblocs-1)/nblocs;
        if (nblocs > 1)
        {
            // round-up bsize to multiples of 16
            bsize = (bsize+15)&-16;
        }
        dim3 threads(bsize,1);
        dim3 grid(nblocs,1);
        int offset;
        if (bsize==1)
            offset = 0;
        else
        {
            offset = 1;
            while (offset*2 < bsize)
                offset *= 2;
        }
        MechanicalObjectCudaVec3f_vDot_kernel<<< grid, threads, bsize * sizeof(float) >>>((float*)tmp, (const float*)a, (const float*)b, size, offset);
        if (nblocs == 1)
        {
            cudaMemcpy(res,tmp,sizeof(float),cudaMemcpyDeviceToHost);
        }
        else
        {
            //float rtmp[nblocs];
            float *rtmp = (float*) alloca(nblocs*sizeof(float));
            cudaMemcpy(rtmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
            float r = 0.0f;
            for (int i=0; i<nblocs; i++)
                r+=rtmp[i];
            *res = r;
        }
    }
}
