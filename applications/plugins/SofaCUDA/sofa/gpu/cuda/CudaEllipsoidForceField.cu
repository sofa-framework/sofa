/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

struct GPUEllipsoid
{
    CudaVec3<float> center;
    CudaVec3<float> inv_r2;
    float stiffness;
    float damping;
};

extern "C"
{
    void EllipsoidForceFieldCuda3f_addForce(unsigned int size, GPUEllipsoid* ellipsoid, float* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const float* tmp, void* f, const void* dx, double factor); //, const void* dfdx);

    void EllipsoidForceFieldCuda3f1_addForce(unsigned int size, GPUEllipsoid* ellipsoid, float* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f1_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const float* tmp, void* f, const void* dx, double factor); //, const void* dfdx);

    int EllipsoidForceFieldCuda3f_getNTmp();
}

//////////////////////
// GPU-side methods //
//////////////////////

#define NTMP 10

int EllipsoidForceFieldCuda3f_getNTmp()
{
    return NTMP;
}

__global__ void EllipsoidForceFieldCuda3f_addForce_kernel(int size, GPUEllipsoid ellipsoid, float* tmp, float* f, const float* x, const float* v)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    tmp += index0*NTMP;
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

    CudaVec3<float> dp = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]) - ellipsoid.center;

    __syncthreads();

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    __syncthreads();

    CudaVec3<float> vi = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    CudaVec3<float> force = CudaVec3<float>::make(0,0,0);

    //float d2 = dot(dp,dp);
    CudaVec3<float> grad = CudaVec3<float>::make(dp.x*ellipsoid.inv_r2.x, dp.y*ellipsoid.inv_r2.y, dp.z*ellipsoid.inv_r2.z);
    //float norm2 = dp.x*dp.x*ellipsoid.inv_r2.x + dp.y*dp.y*ellipsoid.inv_r2.y + dp.z*dp.z*ellipsoid.inv_r2.z;
    float norm2 = dp.x*grad.x + dp.y*grad.y + dp.z*grad.z;
    //Real d = (norm2-1)*s2;
    CudaVec3<float> mx,my,mz;
    float d = (norm2-1)*ellipsoid.stiffness;
    tmp[index+0*BSIZE] = d;
    if (d<0)
    {
        //float norm = sqrt(norm2);
        float inv_norm = rsqrtf(norm2);
        float norm = 1.0f/inv_norm;
        float stiffabs = abs(ellipsoid.stiffness);
        float v = norm-1;
        //for (int j=0; j<N; j++) grad[j] = dp[j]*inv_r2[j];
        float gnorm2 = dot(grad,grad);
        float inv_gnorm = rsqrtf(gnorm2);
        //grad /= gnorm; //.normalize();
        float forceIntensity = -stiffabs*v*inv_gnorm;
        float dampingIntensity = ellipsoid.damping*abs(v);
        force = grad*forceIntensity - vi*dampingIntensity;
        float fact1 = -stiffabs*inv_norm*inv_gnorm;
        float fact2 = -stiffabs*v*inv_gnorm;
        float fact3 = fact2*inv_gnorm; //-stiffabs*v / gnorm2;
        mx.x = grad.x*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x)) + fact2*ellipsoid.inv_r2.x;
        mx.y = grad.x*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y));
        mx.z = grad.x*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z));
        my.x = grad.y*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x));
        my.y = grad.y*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y)) + fact2*ellipsoid.inv_r2.y;
        my.z = grad.y*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z));
        mz.x = grad.z*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x));
        mz.y = grad.z*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y));
        mz.z = grad.z*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z)) + fact2*ellipsoid.inv_r2.z;
    }
    tmp[index+1*BSIZE] = mx.x;
    tmp[index+2*BSIZE] = mx.y;
    tmp[index+3*BSIZE] = mx.z;
    tmp[index+4*BSIZE] = my.x;
    tmp[index+5*BSIZE] = my.y;
    tmp[index+6*BSIZE] = my.z;
    tmp[index+7*BSIZE] = mz.x;
    tmp[index+8*BSIZE] = mz.y;
    tmp[index+9*BSIZE] = mz.z;

    __syncthreads();

    temp[index_3+0] = force.x;
    temp[index_3+1] = force.y;
    temp[index_3+2] = force.z;

    __syncthreads();

    f[index        ] += temp[index        ];
    f[index+  BSIZE] += temp[index+  BSIZE];
    f[index+2*BSIZE] += temp[index+2*BSIZE];
}

__global__ void EllipsoidForceFieldCuda3f1_addForce_kernel(int size, GPUEllipsoid ellipsoid, float* tmp, CudaVec4<float>* f, const CudaVec4<float>* x, const CudaVec4<float>* v)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    tmp += umul24(blockIdx.x,BSIZE*NTMP);

    CudaVec4<float> temp = x[index];
    CudaVec3<float> dp = CudaVec3<float>::make(temp) - ellipsoid.center;
    //float d2 = dot(dp,dp);

    CudaVec4<float> vi = v[index];
    CudaVec3<float> force = CudaVec3<float>::make(0,0,0);

    CudaVec3<float> grad = CudaVec3<float>::make(dp.x*ellipsoid.inv_r2.x, dp.y*ellipsoid.inv_r2.y, dp.z*ellipsoid.inv_r2.z);
    //float norm2 = dp.x*dp.x*ellipsoid.inv_r2.x + dp.y*dp.y*ellipsoid.inv_r2.y + dp.z*dp.z*ellipsoid.inv_r2.z;
    float norm2 = dp.x*grad.x + dp.y*grad.y + dp.z*grad.z;
    //float d = (norm2-1)*s2;
    CudaVec3<float> mx,my,mz;
    float d = (norm2-1)*ellipsoid.stiffness;
    tmp[threadIdx.x+0*BSIZE] = d;
    if (d<0)
    {
        //float norm = sqrt(norm2);
        float inv_norm = rsqrtf(norm2);
        float norm = 1.0f/inv_norm;
        float stiffabs = abs(ellipsoid.stiffness);
        float v = norm-1;
        //for (int j=0; j<N; j++) grad[j] = dp[j]*inv_r2[j];
        float gnorm2 = dot(grad,grad);
        float inv_gnorm = rsqrtf(gnorm2);
        //grad /= gnorm; //.normalize();
        float forceIntensity = -stiffabs*v*inv_gnorm;
        float dampingIntensity = ellipsoid.damping*abs(v);
        force = grad*forceIntensity - CudaVec3<float>::make(vi)*dampingIntensity;
        float fact1 = -stiffabs*inv_norm*inv_gnorm;
        float fact2 = -stiffabs*v*inv_gnorm;
        float fact3 = fact2*inv_gnorm; //-stiffabs*v / gnorm2;
        mx.x = grad.x*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x)) + fact2*ellipsoid.inv_r2.x;
        mx.y = grad.x*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y));
        mx.z = grad.x*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z));
        my.x = grad.y*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x));
        my.y = grad.y*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y)) + fact2*ellipsoid.inv_r2.y;
        my.z = grad.y*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z));
        mz.x = grad.z*(grad.x*(fact1+fact3*ellipsoid.inv_r2.x));
        mz.y = grad.z*(grad.y*(fact1+fact3*ellipsoid.inv_r2.y));
        mz.z = grad.z*(grad.z*(fact1+fact3*ellipsoid.inv_r2.z)) + fact2*ellipsoid.inv_r2.z;
    }
    tmp[threadIdx.x+1*BSIZE] = mx.x;
    tmp[threadIdx.x+2*BSIZE] = mx.y;
    tmp[threadIdx.x+3*BSIZE] = mx.z;
    tmp[threadIdx.x+4*BSIZE] = my.x;
    tmp[threadIdx.x+5*BSIZE] = my.y;
    tmp[threadIdx.x+6*BSIZE] = my.z;
    tmp[threadIdx.x+7*BSIZE] = mz.x;
    tmp[threadIdx.x+8*BSIZE] = mz.y;
    tmp[threadIdx.x+9*BSIZE] = mz.z;

    temp = f[index];
    temp.x += force.x;
    temp.y += force.y;
    temp.z += force.z;
    f[index] = temp;
}

__global__ void EllipsoidForceFieldCuda3f_addDForce_kernel(int size, const float* tmp, float* df, const float* dx, float factor)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    tmp += index0*NTMP;
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
    float d = tmp[index+0*BSIZE];
    CudaVec3<float> mx,my,mz;
    mx.x = tmp[index+1*BSIZE];
    mx.y = tmp[index+2*BSIZE];
    mx.z = tmp[index+3*BSIZE];
    my.x = tmp[index+4*BSIZE];
    my.y = tmp[index+5*BSIZE];
    my.z = tmp[index+6*BSIZE];
    mz.x = tmp[index+7*BSIZE];
    mz.y = tmp[index+8*BSIZE];
    mz.z = tmp[index+9*BSIZE];

    CudaVec3<float> dforce = CudaVec3<float>::make(0,0,0);

    if (d<0)
    {
        dforce.x = dot(mx,dxi)*factor;
        dforce.y = dot(my,dxi)*factor;
        dforce.z = dot(mz,dxi)*factor;
    }

    __syncthreads();

    temp[index_3+0] = dforce.x;
    temp[index_3+1] = dforce.y;
    temp[index_3+2] = dforce.z;

    __syncthreads();

    df[index        ] += temp[index        ];
    df[index+  BSIZE] += temp[index+  BSIZE];
    df[index+2*BSIZE] += temp[index+2*BSIZE];
}

__global__ void EllipsoidForceFieldCuda3f1_addDForce_kernel(int size, const float* tmp, CudaVec4<float>* df, const CudaVec4<float>* dx, float factor)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    tmp += umul24(blockIdx.x,BSIZE*NTMP);

    CudaVec4<float> dxi = dx[index];
    float d = tmp[threadIdx.x+0*BSIZE];
    CudaVec3<float> mx,my,mz;
    mx.x = tmp[threadIdx.x+1*BSIZE];
    mx.y = tmp[threadIdx.x+2*BSIZE];
    mx.z = tmp[threadIdx.x+3*BSIZE];
    my.x = tmp[threadIdx.x+4*BSIZE];
    my.y = tmp[threadIdx.x+5*BSIZE];
    my.z = tmp[threadIdx.x+6*BSIZE];
    mz.x = tmp[threadIdx.x+7*BSIZE];
    mz.y = tmp[threadIdx.x+8*BSIZE];
    mz.z = tmp[threadIdx.x+9*BSIZE];

    CudaVec3<float> dforce = CudaVec3<float>::make(0,0,0);

    if (d<0)
    {
        dforce.x = dot(mx,CudaVec3<float>::make(dxi))*factor;
        dforce.y = dot(my,CudaVec3<float>::make(dxi))*factor;
        dforce.z = dot(mz,CudaVec3<float>::make(dxi))*factor;
    }

    CudaVec4<float> dfi = df[index];
    dfi.x += dforce.x;
    dfi.y += dforce.y;
    dfi.y += dforce.z;
    df[index] = dfi;
}

//////////////////////
// CPU-side methods //
//////////////////////

void EllipsoidForceFieldCuda3f_addForce(unsigned int size, GPUEllipsoid* ellipsoid, float* tmp, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {EllipsoidForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, *ellipsoid, tmp, (float*)f, (const float*)x, (const float*)v); mycudaDebugError("EllipsoidForceFieldCuda3f_addForce_kernel");}
}

void EllipsoidForceFieldCuda3f1_addForce(unsigned int size, GPUEllipsoid* ellipsoid, float* tmp, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {EllipsoidForceFieldCuda3f1_addForce_kernel<<< grid, threads >>>(size, *ellipsoid, tmp, (CudaVec4<float>*)f, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v); mycudaDebugError("EllipsoidForceFieldCuda3f1_addForce_kernel");}
}

void EllipsoidForceFieldCuda3f_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const float* tmp, void* df, const void* dx, double factor) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {EllipsoidForceFieldCuda3f_addDForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, /* *ellipsoid, */ tmp, (float*)df, (const float*)dx, (float)factor); mycudaDebugError("EllipsoidForceFieldCuda3f_addDForce_kernel");}
}

void EllipsoidForceFieldCuda3f1_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const float* tmp, void* df, const void* dx, double factor) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {EllipsoidForceFieldCuda3f1_addDForce_kernel<<< grid, threads >>>(size, /* *ellipsoid, */ tmp, (CudaVec4<float>*)df, (const CudaVec4<float>*)dx, (float)factor); mycudaDebugError("EllipsoidForceFieldCuda3f1_addDForce_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
