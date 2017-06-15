/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

struct GPUSphere
{
    CudaVec3<float> center;
    float r;
    float stiffness;
    float damping;
};

extern "C"
{
    void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, CudaVec4<float>* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const CudaVec4<float>* penetration, void* f, const void* dx); //, const void* dfdx);

    void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, CudaVec4<float>* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const CudaVec4<float>* penetration, void* f, const void* dx); //, const void* dfdx);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void SphereForceFieldCuda3f_addForce_kernel(int size, GPUSphere sphere, CudaVec4<float>* penetration, float* f, const float* x, const float* v)
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

    CudaVec3<float> dp = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]) - sphere.center;
    float d2 = dot(dp,dp);

    __syncthreads();

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    __syncthreads();

    CudaVec3<float> vi = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    CudaVec3<float> force = CudaVec3<float>::make(0,0,0);

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
    penetration[index] = CudaVec4<float>::make(dp.x,dp.y,dp.z,d2);

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

__global__ void SphereForceFieldCuda3f1_addForce_kernel(int size, GPUSphere sphere, CudaVec4<float>* penetration, CudaVec4<float>* f, const CudaVec4<float>* x, const CudaVec4<float>* v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaVec4<float> temp = x[index];
    CudaVec3<float> dp = CudaVec3<float>::make(temp) - sphere.center;
    float d2 = dot(dp,dp);

    CudaVec4<float> vi = v[index];
    CudaVec3<float> force = CudaVec3<float>::make(0,0,0);

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
        force = dp*forceIntensity - CudaVec3<float>::make(vi)*dampingIntensity;
    }
    penetration[index] = CudaVec4<float>::make(dp.x,dp.y,dp.z,d2);

    temp = f[index];
    temp.x += force.x;
    temp.y += force.y;
    temp.z += force.z;
    f[index] = temp;
}

__global__ void SphereForceFieldCuda3f_addDForce_kernel(int size, GPUSphere sphere, const CudaVec4<float>* penetration, float* df, const float* dx)
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

    CudaVec3<float> dxi = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    CudaVec4<float> d = penetration[index];

    CudaVec3<float> dforce = CudaVec3<float>::make(0,0,0);

    if (d.w<0)
    {
        CudaVec3<float> dp = CudaVec3<float>::make(d.x, d.y, d.z);
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

__global__ void SphereForceFieldCuda3f1_addDForce_kernel(int size, GPUSphere sphere, const CudaVec4<float>* penetration, CudaVec4<float>* df, const CudaVec4<float>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaVec4<float> dxi = dx[index];
    CudaVec4<float> d = penetration[index];

    CudaVec3<float> dforce = CudaVec3<float>::make(0,0,0);

    if (d.w<0)
    {
        CudaVec3<float> dp = CudaVec3<float>::make(d.x, d.y, d.z);
        dforce = sphere.stiffness*(dot(CudaVec3<float>::make(dxi),dp)*d.w * dp - (1+d.w) * CudaVec3<float>::make(dxi));
    }

    __syncthreads();

    CudaVec4<float> dfi = df[index];
    dfi.x += dforce.x;
    dfi.y += dforce.y;
    dfi.y += dforce.z;
    df[index] = dfi;
}

//////////////////////
// CPU-side methods //
//////////////////////

void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, CudaVec4<float>* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SphereForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *sphere, penetration, (float*)f, (const float*)x, (const float*)v); mycudaDebugError("SphereForceFieldCuda3f_addForce_kernel");}
}

void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, CudaVec4<float>* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SphereForceFieldCuda3f1_addForce_kernel<<< grid, threads >>>(size, *sphere, penetration, (CudaVec4<float>*)f, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v); mycudaDebugError("SphereForceFieldCuda3f1_addForce_kernel");}
}

void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const CudaVec4<float>* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SphereForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *sphere, penetration, (float*)df, (const float*)dx); mycudaDebugError("SphereForceFieldCuda3f_addDForce_kernel");}
}

void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const CudaVec4<float>* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SphereForceFieldCuda3f1_addDForce_kernel<<< grid, threads >>>(size, *sphere, penetration, (CudaVec4<float>*)df, (const CudaVec4<float>*)dx); mycudaDebugError("SphereForceFieldCuda3f1_addDForce_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
