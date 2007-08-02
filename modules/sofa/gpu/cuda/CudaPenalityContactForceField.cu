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
    void PenalityContactForceFieldCuda3f_setContacts(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* outputs, void* contacts, float d0, float stiffness, matrix3 xform);
    void PenalityContactForceFieldCuda3f_addForce(unsigned int size, const void* contacts, void* pen, void* f1, const void* x1, const void* v1, void* f2, const void* x2, const void* v2);
    void PenalityContactForceFieldCuda3f_addDForce(unsigned int size, const void* contacts, const void* pen, void* f1, const void* dx1, void* f2, const void* dx2);
}

struct /*__align__(16)*/ GPUContact
{
    int p1;
    float3 p2;
    float distance;
    float3 normal;
};

struct /*__align__(8)*/ GPUTestEntry
{
    int firstIndex;
    int curSize;
    int maxSize;
    int newIndex;
    int elem1,elem2;
};

//////////////////////
// GPU-side methods //
//////////////////////

__shared__ GPUTestEntry curTestEntry;

__global__ void PenalityContactForceFieldCuda3f_setContacts_kernel(const GPUTestEntry* tests, const GPUContact* outputs, float4* contacts, float d0, float stiffness, matrix3 xform)
{
    if (threadIdx.x == 0)
        curTestEntry = tests[blockIdx.x];

    __syncthreads();

    GPUContact c = outputs[curTestEntry.firstIndex + threadIdx.x];
    if (threadIdx.x < curTestEntry.curSize)
    {
        float3 n = xform * c.normal;
        //float3 n = xform * make_float3(0,0,-1); //c.normal;
        float d = c.distance + d0;
        //float ks = sqrt(stiffness / d);
        float ks = rsqrt(d / (stiffness*0.000001));
        n *= ks;
        d *= ks;
        contacts[curTestEntry.newIndex + threadIdx.x] =  make_float4(n.x,n.y,n.z,d);
    }
}

__global__ void PenalityContactForceFieldCuda3f_addForce_kernel(int size, const float4* contacts, float* pen, float* f1, const float* x1, const float* v1, float* f2, const float* x2, const float* v2)
//, GPUSphere sphere, float4* penetration, float* f, const float* x, const float* v)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    contacts += index0;
    pen += index0;
    f1 += index0_3;
    x1 += index0_3;
    v1 += index0_3;
    f2 += index0_3;
    x2 += index0_3;
    v2 += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    temp[index        ] = x2[index        ]-x1[index        ];
    temp[index+  BSIZE] = x2[index+  BSIZE]-x1[index+  BSIZE];
    temp[index+2*BSIZE] = x2[index+2*BSIZE]-x1[index+2*BSIZE];

    __syncthreads();

    float3 u = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float4 c = contacts[index];
    float p = c.w - (u.x*c.x+u.y*c.y+u.z*c.z);
    //pen[index] = p;
    float3 force = make_float3(0,0,0);
    if (p>0)
    {
        force.x = -c.x*p;
        force.y = -c.y*p;
        force.z = -c.z*p;
    }

    temp[index_3  ] = force.x;
    temp[index_3+1] = force.y;
    temp[index_3+2] = force.z;

    __syncthreads();
    /*
    f1[index        ] += temp[index        ];
    f1[index+  BSIZE] += temp[index+  BSIZE];
    f1[index+2*BSIZE] += temp[index+2*BSIZE];

    f2[index        ] -= temp[index        ];
    f2[index+  BSIZE] -= temp[index+  BSIZE];
    f2[index+2*BSIZE] -= temp[index+2*BSIZE];*/
}

__global__ void PenalityContactForceFieldCuda3f_addDForce_kernel(int size, const float4* contacts, const float* pen, float* df1, const float* dx1, float* df2, const float* dx2)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    contacts += index0;
    pen += index0;
    df1 += index0_3;
    dx1 += index0_3;
    df2 += index0_3;
    dx2 += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    temp[index        ] = dx2[index        ]-dx1[index        ];
    temp[index+  BSIZE] = dx2[index+  BSIZE]-dx1[index+  BSIZE];
    temp[index+2*BSIZE] = dx2[index+2*BSIZE]-dx1[index+2*BSIZE];

    __syncthreads();

    float3 du = make_float3(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float4 c = contacts[index];
    float3 force = make_float3(0,0,0);
    if (pen[index]>0)
    {
        float dp = - (du.x*c.x+du.y*c.y+du.z*c.z);
        force.x = -c.x*dp;
        force.y = -c.y*dp;
        force.z = -c.z*dp;
    }

    temp[index_3+0] = force.x;
    temp[index_3+1] = force.y;
    temp[index_3+2] = force.z;

    __syncthreads();
    /*
    df1[index        ] += temp[index        ];
    df1[index+  BSIZE] += temp[index+  BSIZE];
    df1[index+2*BSIZE] += temp[index+2*BSIZE];

    df2[index        ] -= temp[index        ];
    df2[index+  BSIZE] -= temp[index+  BSIZE];
    df2[index+2*BSIZE] -= temp[index+2*BSIZE];*/
}

//////////////////////
// CPU-side methods //
//////////////////////

void PenalityContactForceFieldCuda3f_setContacts(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* outputs, void* contacts, float d0, float stiffness, matrix3 xform)
{
    // round up to 16
    //maxPoints = (maxPoints+15)&-16;
    dim3 threads(maxPoints,1);
    dim3 grid(nbTests,1);
    PenalityContactForceFieldCuda3f_setContacts_kernel<<< grid, threads >>>((const GPUTestEntry*)tests, (GPUContact*)outputs, (float4*)contacts, d0, stiffness, xform);
}

void PenalityContactForceFieldCuda3f_addForce(unsigned int size, const void* contacts, void* pen, void* f1, const void* x1, const void* v1, void* f2, const void* x2, const void* v2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PenalityContactForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const float4*)contacts, (float*)pen, (float*)f1, (const float*)x1, (const float*)v1, (float*)f2, (const float*)x2, (const float*)v2);
}

void PenalityContactForceFieldCuda3f_addDForce(unsigned int size, const void* contacts, const void* pen, void* df1, const void* dx1, void* df2, const void* dx2)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PenalityContactForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const float4*)contacts, (const float*)pen, (float*)df1, (const float*)dx1, (float*)df2, (const float*)dx2);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
