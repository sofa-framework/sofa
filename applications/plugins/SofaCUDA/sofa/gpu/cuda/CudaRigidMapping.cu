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

extern "C"
{
    void RigidMappingCuda3f_apply(unsigned int size, const matrix3<float>& rotation, const CudaVec3<float>& translation, void* out, void* rotated, const void* in);
    void RigidMappingCuda3f_applyJ(unsigned int size, const CudaVec3<float>& v, const CudaVec3<float>& omega, void* out, const void* rotated);
    void RigidMappingCuda3f_applyJT(unsigned int size, unsigned int nbloc, void* out, const void* rotated, const void* in);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void RigidMappingCuda3f_apply_kernel(unsigned int size, CudaVec3<float> rotation_x, CudaVec3<float> rotation_y, CudaVec3<float> rotation_z, CudaVec3<float> translation, float* out, float* rotated, const float* in)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    int base = umul24(index0,3);
    in  += base;
    out += base;
    rotated += base;

    temp[index1        ] = in[index1        ];
    temp[index1+  BSIZE] = in[index1+  BSIZE];
    temp[index1+2*BSIZE] = in[index1+2*BSIZE];

    __syncthreads();

    int index3 = umul24(3,index1);
    CudaVec3<float> p = CudaVec3<float>::make(temp[index3  ],temp[index3+1],temp[index3+2]);

    // rotated
    //p = rotation*p;

    temp[index3  ] = dot(rotation_x,p);
    temp[index3+1] = dot(rotation_y,p);
    temp[index3+2] = dot(rotation_z,p);

    __syncthreads();

    rotated[index1        ] = temp[index1        ];
    rotated[index1+  BSIZE] = temp[index1+  BSIZE];
    rotated[index1+2*BSIZE] = temp[index1+2*BSIZE];

    __syncthreads();

    temp[index3  ] += translation.x;
    temp[index3+1] += translation.y;
    temp[index3+2] += translation.z;

    __syncthreads();

    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

__global__ void RigidMappingCuda3f_applyJ_kernel(unsigned int size, CudaVec3<float> v, CudaVec3<float> omega, float* out, const float* rotated)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    int base = umul24(index0,3);
    out += base;
    rotated += base;

    temp[index1        ] = rotated[index1        ];
    temp[index1+  BSIZE] = rotated[index1+  BSIZE];
    temp[index1+2*BSIZE] = rotated[index1+2*BSIZE];

    __syncthreads();

    int index3 = umul24(3,index1);
    CudaVec3<float> p = v - cross(CudaVec3<float>::make(temp[index3  ],temp[index3+1],temp[index3+2]),omega);

    temp[index3  ] = p.x;
    temp[index3+1] = p.y;
    temp[index3+2] = p.z;

    __syncthreads();

    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

__global__ void RigidMappingCuda3f_applyJT_kernel(unsigned int size, unsigned int nbloc, float* out, const float* rotated, const float* in)
{
    int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> t = CudaVec3<float>::make(0.0f, 0.0f, 0.0f);
    CudaVec3<float> r = CudaVec3<float>::make(0.0f, 0.0f, 0.0f);

    int index3 = umul24(3,index1);

    while (index0 < size)
    {

        int base = umul24(index0,3);

        temp[index1        ] = in[base+index1        ];
        temp[index1+  BSIZE] = in[base+index1+  BSIZE];
        temp[index1+2*BSIZE] = in[base+index1+2*BSIZE];
        temp[index1+3*BSIZE] = rotated[base+index1        ];
        temp[index1+4*BSIZE] = rotated[base+index1+  BSIZE];
        temp[index1+5*BSIZE] = rotated[base+index1+2*BSIZE];

        __syncthreads();

        if (index0+index1 < size)
        {
            CudaVec3<float> v = CudaVec3<float>::make(temp[index3  ],temp[index3+1],temp[index3+2]);
            t += v;
            r += cross(CudaVec3<float>::make(temp[index3  +3*BSIZE],temp[index3+1+3*BSIZE],temp[index3+2+3*BSIZE]),v);
        }
        __syncthreads();
        index0 += umul24(nbloc, BSIZE);
    }

    temp[index3  ] = t.x;
    temp[index3+1] = t.y;
    temp[index3+2] = t.z;
    temp[index3  +3*BSIZE] = r.x;
    temp[index3+1+3*BSIZE] = r.y;
    temp[index3+2+3*BSIZE] = r.z;

    //__syncthreads();

    int offset = BSIZE/2;
    int offset3 = (BSIZE/2) * 3;

    while(offset>0)
    {
        //if (index1 >= offset && index1 < BSIZE)
        //	    temp[index1] = acc;
        __syncthreads();
        if (index1 < offset)
        {
            temp[index3  ]+=temp[index3+offset3  ];
            temp[index3+1]+=temp[index3+offset3+1];
            temp[index3+2]+=temp[index3+offset3+2];
            temp[index3  +3*BSIZE]+=temp[index3+offset3  +3*BSIZE];
            temp[index3+1+3*BSIZE]+=temp[index3+offset3+1+3*BSIZE];
            temp[index3+2+3*BSIZE]+=temp[index3+offset3+2+3*BSIZE];
        }
        offset >>= 1;
        offset3 >>= 1;
    }
    __syncthreads();
    if (index1 < 6)
    {
        out[umul24(blockIdx.x,6) + index1] = temp[index1];
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void RigidMappingCuda3f_apply(unsigned int size, const matrix3<float>& rotation, const CudaVec3<float>& translation, void* out, void* rotated, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RigidMappingCuda3f_apply_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, rotation.x, rotation.y, rotation.z, translation, (float*)out, (float*)rotated, (const float*)in); mycudaDebugError("RigidMappingCuda3f_apply_kernel");}
}

void RigidMappingCuda3f_applyJ(unsigned int size, const CudaVec3<float>& v, const CudaVec3<float>& omega, void* out, const void* rotated)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RigidMappingCuda3f_applyJ_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, v, omega, (float*)out, (const float*)rotated); mycudaDebugError("RigidMappingCuda3f_applyJ_kernel");}
}

void RigidMappingCuda3f_applyJT(unsigned int size, unsigned int nbloc, void* out, const void* rotated, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid(nbloc,1);
    {RigidMappingCuda3f_applyJT_kernel<<< grid, threads, BSIZE*6*sizeof(float) >>>(size, nbloc, (float*)out, (const float*)rotated, (const float*)in); mycudaDebugError("RigidMappingCuda3f_applyJT_kernel");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
