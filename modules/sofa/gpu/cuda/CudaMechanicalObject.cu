#include "CudaCommon.h"
#include "CudaMath.h"
#include "mycuda.h"
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
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vClear(unsigned int size, void* res);
    void MechanicalObjectCudaVec3f_vMEq(unsigned int size, void* res, float f);
    void MechanicalObjectCudaVec3f_vEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEqBF(unsigned int size, void* res, const void* b, float f);
    void MechanicalObjectCudaVec3f_vAdd(unsigned int size, void* res, const void* a, const void* b);
    void MechanicalObjectCudaVec3f_vOp(unsigned int size, void* res, const void* a, const void* b, float f);
    void MechanicalObjectCudaVec3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
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
    void MechanicalObjectCudaVec3f1_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v);
    void MechanicalObjectCudaVec3f1_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2);
    void MechanicalObjectCudaVec3f1_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
            void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24);
    void MechanicalObjectCudaVec3f1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2);
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

__global__ void MechanicalObjectCudaVec1f_vPEqBF2_kernel(int size, float* res1, const float* b1, float f1, float* res2, const float* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
    }
}

__global__ void MechanicalObjectCudaVec3f_vPEqBF2_kernel(int size, float* res1, const float* b1, float f1, float* res2, const float* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        index += BSIZE;
        res1[index] += b1[index] * f1;
        res2[index] += b2[index] * f2;
        //float3 bi = b[index];
        //float3 ri = res[index];
        //ri += bi * f;
        //res[index] = ri;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vPEqBF2_kernel(int size, float4* res1, const float4* b1, float f1, float4* res2, const float4* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] += b[index] * f;
        float4 v = res1[index];
        float4 v2 = b1[index];
        v.x += v2.x*f1;
        v.y += v2.y*f1;
        v.z += v2.z*f1;
        res1[index] = v;
        v = res2[index];
        v2 = b2[index];
        v.x += v2.x*f2;
        v.y += v2.y*f2;
        v.z += v2.z*f2;
        res2[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vPEq4BF2_kernel(int size, float* res1, const float* b11, float f11, const float* b12, float f12, const float* b13, float f13, const float* b14, float f14,
        float* res2, const float* b21, float f21, const float* b22, float f22, const float* b23, float f23, const float* b24, float f24)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        float r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}

__global__ void MechanicalObjectCudaVec3f_vPEq4BF2_kernel(int size, float* res1, const float* b11, float f11, const float* b12, float f12, const float* b13, float f13, const float* b14, float f14,
        float* res2, const float* b21, float f21, const float* b22, float f22, const float* b23, float f23, const float* b24, float f24)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        float r1,r2;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
        index += BSIZE;
        r1 = res1[index];
        r2 = res2[index];
        r1 += b11[index] * f11;
        r2 += b21[index] * f21;
        r1 += b12[index] * f12;
        r2 += b22[index] * f22;
        r1 += b13[index] * f13;
        r2 += b23[index] * f23;
        r1 += b14[index] * f14;
        r2 += b24[index] * f24;
        res1[index] = r1;
        res2[index] = r2;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vPEq4BF2_kernel(int size, float4* res1, const float4* b11, float f11, const float4* b12, float f12, const float4* b13, float f13, const float4* b14, float f14,
        float4* res2, const float4* b21, float f21, const float4* b22, float f22, const float4* b23, float f23, const float4* b24, float f24)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        float4 v = res1[index];
        float4 v2 = b11[index];
        v.x += v2.x*f11;
        v.y += v2.y*f11;
        v.z += v2.z*f11;
        v2 = b12[index];
        v.x += v2.x*f12;
        v.y += v2.y*f12;
        v.z += v2.z*f12;
        v2 = b13[index];
        v.x += v2.x*f13;
        v.y += v2.y*f13;
        v.z += v2.z*f13;
        v2 = b14[index];
        v.x += v2.x*f14;
        v.y += v2.y*f14;
        v.z += v2.z*f14;
        res1[index] = v;
        v = res2[index];
        v2 = b21[index];
        v.x += v2.x*f21;
        v.y += v2.y*f21;
        v.z += v2.z*f21;
        v2 = b22[index];
        v.x += v2.x*f22;
        v.y += v2.y*f22;
        v.z += v2.z*f22;
        v2 = b23[index];
        v.x += v2.x*f23;
        v.y += v2.y*f23;
        v.z += v2.z*f23;
        v2 = b24[index];
        v.x += v2.x*f24;
        v.y += v2.y*f24;
        v.z += v2.z*f24;
        res2[index] = v;
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


__global__ void MechanicalObjectCudaVec1f_vOp2_kernel(int size, float* res1, const float* a1, const float* b1, float f1, float* res2, const float* a2, const float* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}

__global__ void MechanicalObjectCudaVec3f_vOp2_kernel(int size, float* res1, const float* a1, const float* b1, float f1, float* res2, const float* a2, const float* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
        index += BSIZE;
        res1[index] = a1[index] + b1[index] * f1;
        res2[index] = a2[index] + b2[index] * f2;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vOp2_kernel(int size, float4* res1, const float4* a1, const float4* b1, float f1, float4* res2, const float4* a2, const float4* b2, float f2)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        float4 v = a1[index];
        float4 v2 = b1[index];
        v.x += v2.x*f1;
        v.y += v2.y*f1;
        v.z += v2.z*f1;
        res1[index] = v;
        v = a2[index];
        v2 = b2[index];
        v.x += v2.x*f2;
        v.y += v2.y*f2;
        v.z += v2.z*f2;
        res2[index] = v;
    }
}

__global__ void MechanicalObjectCudaVec1f_vIntegrate_kernel(int size, const float* a, float* v, float* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        float vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }
}

__global__ void MechanicalObjectCudaVec3f_vIntegrate_kernel(int size, const float* a, float* v, float* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE*3)+threadIdx.x;
    //if (index < size)
    {
        float vi;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
        index += BSIZE;
        vi = v[index]*f_v_v + a[index] * f_v_a;
        v[index] = vi;
        x[index] = x[index]*f_x_x + vi * f_x_v;
    }
}

__global__ void MechanicalObjectCudaVec3f1_vIntegrate_kernel(int size, const float4* a, float4* v, float4* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //if (index < size)
    {
        //res[index] = a[index] + b[index] * f;
        float4 ai = a[index];
        float4 vi = v[index];
        float4 xi = x[index];
        vi.x = vi.x*f_v_v + ai.x*f_v_a;
        vi.y = vi.y*f_v_v + ai.y*f_v_a;
        vi.z = vi.z*f_v_v + ai.z*f_v_a;
        xi.x = xi.x*f_x_x + vi.x*f_x_v;
        xi.y = xi.y*f_x_x + vi.y*f_x_v;
        xi.z = xi.z*f_x_x + vi.z*f_x_v;
        v[index] = vi;
        x[index] = xi;
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
    unsigned int gridSize = gridDim.x*(blockSize);
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += a[i] * b[i]; i += gridSize; }
    __syncthreads();
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32)
    {
#if blockSize >= 64
        //if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
#endif
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
#if blockSize >= 512
    //if (blockSize >= 512)
    {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
#endif
#if blockSize >= 256
    //if (blockSize >= 256)
    {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
#endif
#if blockSize >= 128
    //if (blockSize >= 128)
    {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
#endif
    if (tid < 32)
    {
#if blockSize >= 64
        //if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
#endif
#if blockSize >= 32
        //if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
#endif
#if blockSize >= 16
        //if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
#endif
#if blockSize >= 8
        //if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
#endif
#if blockSize >= 4
        //if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
#endif
#if blockSize >= 2
        //if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
#endif
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

void MechanicalObjectCudaVec3f_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vPEqBF2_kernel<<< grid, threads >>>(size, (float*)res1, (const float*)b1, f1, (float*)res2, (const float*)b2, f2);
}

void MechanicalObjectCudaVec3f1_vPEqBF2(unsigned int size, void* res1, const void* b1, float f1, void* res2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vPEqBF2_kernel<<< grid, threads >>>(size, (float4*)res1, (const float4*)b1, f1, (float4*)res2, (const float4*)b2, f2);
}

void MechanicalObjectCudaVec3f_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vPEq4BF2_kernel<<< grid, threads >>>(size, (float*)res1, (const float*)b11, f11, (const float*)b12, f12, (const float*)b13, f13, (const float*)b14, f14,
            (float*)res2, (const float*)b21, f21, (const float*)b22, f22, (const float*)b23, f23, (const float*)b24, f24);
}

void MechanicalObjectCudaVec3f1_vPEq4BF2(unsigned int size, void* res1, const void* b11, float f11, const void* b12, float f12, const void* b13, float f13, const void* b14, float f14,
        void* res2, const void* b21, float f21, const void* b22, float f22, const void* b23, float f23, const void* b24, float f24)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vPEq4BF2_kernel<<< grid, threads >>>(size, (float4*)res1, (const float4*)b11, f11, (const float4*)b12, f12, (const float4*)b13, f13, (const float4*)b14, f14,
            (float4*)res2, (const float4*)b21, f21, (const float4*)b22, f22, (const float4*)b23, f23, (const float4*)b24, f24);
}

void MechanicalObjectCudaVec3f_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vOp2_kernel<<< grid, threads >>>(size, (float*)res1, (const float*)a1, (const float*)b1, f1, (float*)res2, (const float*)a2, (const float*)b2, f2);
}

void MechanicalObjectCudaVec3f1_vOp2(unsigned int size, void* res1, const void* a1, const void* b1, float f1, void* res2, const void* a2, const void* b2, float f2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vOp2_kernel<<< grid, threads >>>(size, (float4*)res1, (const float4*)a1, (const float4*)b1, f1, (float4*)res2, (const float4*)a2, (const float4*)b2, f2);
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

void MechanicalObjectCudaVec3f_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f_vIntegrate_kernel<<< grid, threads >>>(size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vIntegrate_kernel<<< grid, threads >>>(3*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}

void MechanicalObjectCudaVec3f1_vIntegrate(unsigned int size, const void* a, void* v, void* x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    MechanicalObjectCudaVec3f1_vIntegrate_kernel<<< grid, threads >>>(size, (const float4*)a, (float4*)v, (float4*)x, f_v_v, f_v_a, f_x_x, f_x_v);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //MechanicalObjectCudaVec1f_vIntegrate_kernel<<< grid, threads >>>(4*size, (const float*)a, (float*)v, (float*)x, f_v_v, f_v_a, f_x_x, f_x_v);
}


int MechanicalObjectCudaVec3f_vDotTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+RED_BSIZE-1)/RED_BSIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
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
    return nblocs; //(nblocs+3)/4;
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

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
