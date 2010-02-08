/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
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

template<class real>
class GPUPlane
{
public:
    //CudaVec3<real> normal;
    real normal_x, normal_y, normal_z;
    real d;
    real stiffness;
    real damping;
};

template<class real>
struct PlaneDForceOp
{
    unsigned int size;
    GPUPlane<real> plane;
    const real* penetration;
    real* f;
    const real* dx;
};

typedef GPUPlane<float> GPUPlane3f;
typedef GPUPlane<double> GPUPlane3d;

typedef PlaneDForceOp<float> PlaneDForceOp3f;
typedef PlaneDForceOp<double> PlaneDForceOp3d;

extern "C"
{
    void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane3f* plane, float* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane3f* plane, const float* penetration, void* f, const void* dx); //, const void* dfdx);

    void MultiPlaneForceFieldCuda3f_addDForce(int n, PlaneDForceOp3f* ops);

    void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane3f* plane, float* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane3f* plane, const float* penetration, void* f, const void* dx); //, const void* dfdx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane3d* plane, double* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane3d* plane, const double* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane3d* plane, double* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane3d* plane, const double* penetration, void* f, const void* dx); //, const void* dfdx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void PlaneForceFieldCuda3t_addForce_kernel(int size, GPUPlane<real> plane, real* penetration, real* f, const real* x, const real* v)
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
    __shared__  real temp[BSIZE*3];

    temp[index        ] = x[index        ];
    temp[index+  BSIZE] = x[index+  BSIZE];
    temp[index+2*BSIZE] = x[index+2*BSIZE];

    __syncthreads();

    CudaVec3<real> xi = CudaVec3<real>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    real d = dot(xi,CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z))-plane.d;

    penetration[index] = d;

    __syncthreads();

    temp[index        ] = v[index        ];
    temp[index+  BSIZE] = v[index+  BSIZE];
    temp[index+2*BSIZE] = v[index+2*BSIZE];

    __syncthreads();

    CudaVec3<real> vi = CudaVec3<real>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    CudaVec3<real> force = CudaVec3<real>::make(0,0,0);

    if (d<0)
    {
        real forceIntensity = -plane.stiffness*d;
        real dampingIntensity = -plane.damping*d;
        force = CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z)*forceIntensity - vi*dampingIntensity;
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

template<class real>
__global__ void PlaneForceFieldCuda3t1_addForce_kernel(int size, GPUPlane<real> plane, real* penetration, CudaVec4<real>* f, const CudaVec4<real>* x, const CudaVec4<real>* v)
{
    int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    CudaVec4<real> xi = x[index];
    real d = dot(CudaVec3<real>::make(xi),CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z))-plane.d;

    penetration[index] = d;

    CudaVec4<real> vi = v[index];
    CudaVec3<real> force = CudaVec3<real>::make(0,0,0);

    if (d<0)
    {
        real forceIntensity = -plane.stiffness*d;
        real dampingIntensity = -plane.damping*d;
        force = CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z)*forceIntensity - CudaVec3<real>::make(vi)*dampingIntensity;
    }

    CudaVec4<real> temp = f[index];
    temp.x += force.x;
    temp.y += force.y;
    temp.z += force.z;
    f[index] = temp;
}

template<class real>
__global__ void PlaneForceFieldCuda3t_addDForce_kernel(int size, GPUPlane<real> plane, const real* penetration, real* df, const real* dx)
{
    int index0 = umul24(blockIdx.x,BSIZE);
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;

    penetration += index0;
    df += index0_3;
    dx += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  real temp[BSIZE*3];

    temp[index        ] = dx[index        ];
    temp[index+  BSIZE] = dx[index+  BSIZE];
    temp[index+2*BSIZE] = dx[index+2*BSIZE];

    __syncthreads();

    CudaVec3<real> dxi = CudaVec3<real>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    real d = penetration[index];

    CudaVec3<real> dforce = CudaVec3<real>::make(0,0,0);

    if (d<0)
    {
        dforce = CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z) * (-plane.stiffness * dot(dxi, CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z)));
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

template<class real>
__global__ void PlaneForceFieldCuda3t1_addDForce_kernel(int size, GPUPlane<real> plane, const real* penetration, CudaVec4<real>* df, const CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    CudaVec4<real> dxi = dx[index];
    real d = penetration[index];

    CudaVec3<real> dforce = CudaVec3<real>::make(0,0,0);

    if (d<0)
    {
        dforce = CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z) * (-plane.stiffness * dot(CudaVec3<real>::make(dxi), CudaVec3<real>::make(plane.normal_x,plane.normal_y,plane.normal_z)));
    }
    CudaVec4<real> dfi = df[index];
    dfi.x += dforce.x;
    dfi.y += dforce.y;
    dfi.y += dforce.z;
    df[index] = dfi;
}

enum { MULTI_NMAX = 64 };
__constant__ PlaneDForceOp3f multiPlaneDForceOps3f[MULTI_NMAX];

__global__ void MultiPlaneForceFieldCuda3f_addDForce_kernel()
{
    int index0 = umul24(blockIdx.x,BSIZE);
    if (index0 >= multiPlaneDForceOps3f[blockIdx.y].size) return;
    int index0_3 = umul24(blockIdx.x,BSIZE*3); //index0*3;
    const float* penetration = multiPlaneDForceOps3f[blockIdx.y].penetration;
    float* df = multiPlaneDForceOps3f[blockIdx.y].f;
    const float* dx = multiPlaneDForceOps3f[blockIdx.y].dx;

    penetration += index0;
    df += index0_3;
    dx += index0_3;

    int index = threadIdx.x;
    int index_3 = umul24(index,3); //index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    __shared__  float temp[BSIZE*3];

    temp[index        ] = dx[index        ];
    temp[index+  BSIZE] = dx[index+  BSIZE];
    temp[index+2*BSIZE] = dx[index+2*BSIZE];

    __syncthreads();

    CudaVec3<float> dxi = CudaVec3<float>::make(temp[index_3  ], temp[index_3+1], temp[index_3+2]);
    float d = penetration[index];

    CudaVec3<float> dforce = CudaVec3<float>::make(0,0,0);

    if (d<0)
    {
        CudaVec3<float> normal = CudaVec3<float>::make(multiPlaneDForceOps3f[blockIdx.y].plane.normal_x,multiPlaneDForceOps3f[blockIdx.y].plane.normal_y,multiPlaneDForceOps3f[blockIdx.y].plane.normal_z);
        dforce = normal * (-multiPlaneDForceOps3f[blockIdx.y].plane.stiffness * dot(dxi, normal));
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

//////////////////////
// CPU-side methods //
//////////////////////

void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane3f* plane, float* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t_addForce_kernel<float><<< grid, threads >>>(size, *plane, penetration, (float*)f, (const float*)x, (const float*)v);
}

void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane3f* plane, float* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t1_addForce_kernel<float><<< grid, threads >>>(size, *plane, penetration, (CudaVec4<float>*)f, (const CudaVec4<float>*)x, (const CudaVec4<float>*)v);
}

void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane3f* plane, const float* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t_addDForce_kernel<float><<< grid, threads >>>(size, *plane, penetration, (float*)df, (const float*)dx);
}

void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane3f* plane, const float* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t1_addDForce_kernel<float><<< grid, threads >>>(size, *plane, penetration, (CudaVec4<float>*)df, (const CudaVec4<float>*)dx);
}

void MultiPlaneForceFieldCuda3f_addDForce(int n, PlaneDForceOp3f* ops)
{
    int totalblocs = 0;
    for (unsigned int i0 = 0; i0 < n; i0 += MULTI_NMAX)
    {
        int n2 = (n-i0 > MULTI_NMAX) ? MULTI_NMAX : n-i0;
        int maxnblocs = 0;
        PlaneDForceOp3f* ops2 = ops + i0;
        for (unsigned int i = 0; i < n2; ++i)
        {
            int nblocs = (ops2[i].size+BSIZE-1)/BSIZE;
            if (nblocs > maxnblocs) maxnblocs = nblocs;
        }
        cudaMemcpyToSymbol(multiPlaneDForceOps3f, ops2, n2 * sizeof(PlaneDForceOp3f), 0, cudaMemcpyHostToDevice);

        dim3 threads(BSIZE,1);
        dim3 grid(maxnblocs,n2);
        MultiPlaneForceFieldCuda3f_addDForce_kernel<<< grid, threads >>>();

        totalblocs += maxnblocs * n2;
    }
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane3d* plane, double* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t_addForce_kernel<double><<< grid, threads >>>(size, *plane, penetration, (double*)f, (const double*)x, (const double*)v);
}

void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane3d* plane, double* penetration, void* f, const void* x, const void* v)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t1_addForce_kernel<double><<< grid, threads >>>(size, *plane, penetration, (CudaVec4<double>*)f, (const CudaVec4<double>*)x, (const CudaVec4<double>*)v);
}

void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane3d* plane, const double* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t_addDForce_kernel<double><<< grid, threads >>>(size, *plane, penetration, (double*)df, (const double*)dx);
}

void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane3d* plane, const double* penetration, void* df, const void* dx) //, const void* dfdx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    PlaneForceFieldCuda3t1_addDForce_kernel<double><<< grid, threads >>>(size, *plane, penetration, (CudaVec4<double>*)df, (const CudaVec4<double>*)dx);
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
