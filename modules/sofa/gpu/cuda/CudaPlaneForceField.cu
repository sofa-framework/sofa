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

struct GPUPlane
{
    float3 normal;
    float d;
    float stiffness;
    float damping;
};

extern "C"
{
    void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane* plane, float* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane* plane, const float* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane* plane, float* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane* plane, const float* penetration, void* f, const void* dx); //, const void* dfdx);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void PlaneForceFieldCuda3f_addForce_kernel(int size, GPUPlane plane, float* penetration, float* f, const float* x, const float* v)
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

    float3 xi = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float d = dot(xi,plane.normal)-plane.d;

    penetration[index] = d;

    __syncthreads();

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    __syncthreads();

    float3 vi = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float3 force = make_float3(0,0,0);

    if (d<0)
    {
        float forceIntensity = -plane.stiffness*d;
        float dampingIntensity = -plane.damping*d;
        force = plane.normal*forceIntensity - vi*dampingIntensity;
    }

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

__global__ void PlaneForceFieldCuda3f1_addForce_kernel(int size, GPUPlane plane, float* penetration, float4* f, const float4* x, const float4* v)
{
    int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    float4 xi = x[index];
    float d = dot(make_float3(xi),plane.normal)-plane.d;

    penetration[index] = d;

    float4 vi = v[index];
    float3 force = make_float3(0,0,0);

    if (d<0)
    {
        float forceIntensity = -plane.stiffness*d;
        float dampingIntensity = -plane.damping*d;
        force = plane.normal*forceIntensity - make_float3(vi)*dampingIntensity;
    }

    float4 temp = f[index];
    temp.x += force.x;
    temp.y += force.y;
    temp.z += force.z;
    f[index] = temp;
}

__global__ void PlaneForceFieldCuda3f_addDForce_kernel(int size, GPUPlane plane, const float* penetration, float* df, const float* dx)
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
    float d = penetration[index];

    float3 dforce = make_float3(0,0,0);

    if (d<0)
    {
        dforce = plane.normal * (-plane.stiffness * dot(dxi, plane.normal));
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

__global__ void PlaneForceFieldCuda3f1_addDForce_kernel(int size, GPUPlane plane, const float* penetration, float4* df, const float4* dx)
{
    int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    float4 dxi = dx[index];
    float d = penetration[index];

    float3 dforce = make_float3(0,0,0);

    if (d<0)
    {
        dforce = plane.normal * (-plane.stiffness * dot(make_float3(dxi), plane.normal));
    }
    float4 dfi = df[index];
    dfi.x += dforce.x;
    dfi.y += dforce.y;
    dfi.y += dforce.z;
    df[index] = dfi;
}

//////////////////////
// CPU-side methods //
//////////////////////

void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane* plane, float* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *plane, penetration, (float*)f, (const float*)x, (const float*)v);
}

void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane* plane, float* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3f1_addForce_kernel<<< grid, threads >>>(size, *plane, penetration, (float4*)f, (const float4*)x, (const float4*)v);
}

void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane* plane, const float* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *plane, penetration, (float*)df, (const float*)dx);
}

void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane* plane, const float* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3f1_addDForce_kernel<<< grid, threads >>>(size, *plane, penetration, (float4*)df, (const float4*)dx);
}

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
