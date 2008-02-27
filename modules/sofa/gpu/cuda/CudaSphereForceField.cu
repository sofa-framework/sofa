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

struct GPUSphere
{
    float3 center;
    float r;
    float stiffness;
    float damping;
};

extern "C"
{
    void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, float4* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const float4* penetration, void* f, const void* dx); //, const void* dfdx);

    void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, float4* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const float4* penetration, void* f, const void* dx); //, const void* dfdx);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void SphereForceFieldCuda3f_addForce_kernel(int size, GPUSphere sphere, float4* penetration, float* f, const float* x, const float* v)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    penetration += index0;
    f += index0_3;
    x += index0_3;
    v += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    temp[index        ] = x[index        ];
    temp[index+  BSIZE] = x[index+  BSIZE];
    temp[index+2*BSIZE] = x[index+2*BSIZE];

    __syncthreads();

    float3 dp = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]) - sphere.center;
    float d2 = dot(dp,dp);

    __syncthreads();

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    __syncthreads();

    float3 vi = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float3 force = make_float3(0,0,0);

    if (d2 < sphere.r*sphere.r)
    {
        float inverseLength = 1/sqrt(d2);
        dp.x *= inverseLength;
        dp.y *= inverseLength;
        dp.z *= inverseLength;
        d2 = -sphere.r*inverseLength;
        float d = sphere.r - __fdividef(1,inverseLength);

        float forceIntensity = sphere.stiffness*d;
        float dampingIntensity = sphere.damping*d;
        force = dp*forceIntensity - vi*dampingIntensity;
    }
    penetration[index] = make_float4(dp.x,dp.y,dp.z,d2);

    __syncthreads();

    temp[index        ] = f[index        ];
    temp[index+  BSIZE] = f[index+  BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    __syncthreads();

    temp[index_3+0] += force.x;
    temp[index_3+1] += force.y;
    temp[index_3+2] += force.z;

    __syncthreads();

    f[index        ] = temp[index        ];
    f[index+  BSIZE] = temp[index+  BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}

__global__ void SphereForceFieldCuda3f1_addForce_kernel(int size, GPUSphere sphere, float4* penetration, float4* f, const float4* x, const float4* v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    float4 temp = x[index];
    float3 dp = make_float3(temp) - sphere.center;
    float d2 = dot(dp,dp);

    float4 vi = v[index];
    float3 force = make_float3(0,0,0);

    if (d2 < sphere.r*sphere.r)
    {
        float inverseLength = 1/sqrt(d2);
        dp.x *= inverseLength;
        dp.y *= inverseLength;
        dp.z *= inverseLength;
        d2 = -sphere.r*inverseLength;
        float d = sphere.r - __fdividef(1,inverseLength);

        float forceIntensity = sphere.stiffness*d;
        float dampingIntensity = sphere.damping*d;
        force = dp*forceIntensity - make_float3(vi)*dampingIntensity;
    }
    penetration[index] = make_float4(dp.x,dp.y,dp.z,d2);

    temp = f[index];
    temp.x += force.x;
    temp.y += force.y;
    temp.z += force.z;
    f[index] = temp;
}

__global__ void SphereForceFieldCuda3f_addDForce_kernel(int size, GPUSphere sphere, const float4* penetration, float* df, const float* dx)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    penetration += index0;
    df += index0_3;
    dx += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    temp[index        ] = dx[index        ];
    temp[index+  BSIZE] = dx[index+  BSIZE];
    temp[index+2*BSIZE] = dx[index+2*BSIZE];

    __syncthreads();

    float3 dxi = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float4 d = penetration[index];

    float3 dforce = make_float3(0,0,0);

    if (d.w<0)
    {
        float3 dp = make_float3(d.x, d.y, d.z);
        dforce = sphere.stiffness*(dot(dxi,dp)*d.w * dp - (1+d.w) * dxi);
    }

    __syncthreads();

    temp[index        ] = df[index        ];
    temp[index+  BSIZE] = df[index+  BSIZE];
    temp[index+2*BSIZE] = df[index+2*BSIZE];

    __syncthreads();

    temp[index_3+0] += dforce.x;
    temp[index_3+1] += dforce.y;
    temp[index_3+2] += dforce.z;

    __syncthreads();

    df[index        ] = temp[index        ];
    df[index+  BSIZE] = temp[index+  BSIZE];
    df[index+2*BSIZE] = temp[index+2*BSIZE];
}

__global__ void SphereForceFieldCuda3f1_addDForce_kernel(int size, GPUSphere sphere, const float4* penetration, float4* df, const float4* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    float4 dxi = dx[index];
    float4 d = penetration[index];

    float3 dforce = make_float3(0,0,0);

    if (d.w<0)
    {
        float3 dp = make_float3(d.x, d.y, d.z);
        dforce = sphere.stiffness*(dot(make_float3(dxi),dp)*d.w * dp - (1+d.w) * make_float3(dxi));
    }

    __syncthreads();

    float4 dfi = df[index];
    dfi.x += dforce.x;
    dfi.y += dforce.y;
    dfi.y += dforce.z;
    df[index] = dfi;
}

//////////////////////
// CPU-side methods //
//////////////////////

void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, float4* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SphereForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *sphere, penetration, (float*)f, (const float*)x, (const float*)v);
}

void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, float4* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SphereForceFieldCuda3f1_addForce_kernel<<< grid, threads >>>(size, *sphere, penetration, (float4*)f, (const float4*)x, (const float4*)v);
}

void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const float4* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SphereForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *sphere, penetration, (float*)df, (const float*)dx);
}

void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const float4* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    SphereForceFieldCuda3f1_addDForce_kernel<<< grid, threads >>>(size, *sphere, penetration, (float4*)df, (const float4*)dx);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
