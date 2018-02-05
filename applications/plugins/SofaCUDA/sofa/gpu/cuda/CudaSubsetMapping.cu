/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
    void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT1(unsigned int insize, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f_3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);
}

//////////////////////
// GPU-side methods //
//////////////////////

template<typename TIn>
__global__ void SubsetMappingCuda3f_apply_kernel(unsigned int size, const int* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);

    int c = map[index0+index1];
    if (index0+index1 < size)
    {
        res = CudaVec3<float>::make(in[c]);
    }

    //__syncthreads();

    const int index3 = umul24(index1,3);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

template<typename TIn>
__global__ void SubsetMappingCuda3f1_apply_kernel(unsigned int size, const int* map, CudaVec4<float>* out, const TIn* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    CudaVec4<float> res = CudaVec4<float>::make(0,0,0,0);

    int c = map[index];
    if (index < size)
    {
        res = CudaVec4<float>::make(in[c]);
    }

    out[index] = res;
}

template<typename TIn>
__global__ void SubsetMappingCuda3f_applyJT_kernel(unsigned int size, unsigned int maxNOut, const int* mapT, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *((const CudaVec3<float>*)in) * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        int data = *mapT;
        mapT+=BSIZE;
        if (data != -1)
            res += CudaVec3<float>::make(in[data]);
    }

    const int index3 = umul24(index1,3);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] += temp[index1        ];
    out[index1+  BSIZE] += temp[index1+  BSIZE];
    out[index1+2*BSIZE] += temp[index1+2*BSIZE];
}

template<typename TIn>
__global__ void SubsetMappingCuda3f1_applyJT_kernel(unsigned int size, unsigned int maxNOut, const int* mapT, CudaVec4<float>* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *((const CudaVec3<float>*)in) * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        int data = *mapT;
        mapT+=BSIZE;
        if (data != -1)
            res += CudaVec3<float>::make(in[data]);
    }
    CudaVec4<float> o = out[index];
    o.x += res.x;
    o.y += res.y;
    o.z += res.z;
    out[index] = o;
}

template<typename TOut>
__global__ void SubsetMappingCuda3f_applyJT1_kernel(unsigned int size, const int* map, TOut* out, const float* in)
{

    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    in += umul24(index0,3);
    temp[index1        ] = in[index1        ];
    temp[index1+  BSIZE] = in[index1+  BSIZE];
    temp[index1+2*BSIZE] = in[index1+2*BSIZE];

    __syncthreads();

    const int index3 = umul24(3,index1);
    CudaVec3<float> res = CudaVec3<float>::make(temp[index3  ],temp[index3+1],temp[index3+2]);

    int c = map[index0+index1];
    if (index0+index1 < size)
    {
        TOut o = out[c];
        o.x += res.x;
        o.y += res.y;
        o.z += res.z;
        out[c] = o;
    }
}

template<typename TOut>
__global__ void SubsetMappingCuda3f1_applyJT1_kernel(unsigned int size, const int* map, TOut* out, const CudaVec4<float>* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    CudaVec3<float> res = CudaVec3<float>::make(in[index]);

    int c = map[index];
    if (index < size)
    {
        TOut o = out[c];
        o.x += res.x;
        o.y += res.y;
        o.z += res.z;
        out[c] = o;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f_apply_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f_apply_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_applyJT_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const int*)mapT, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f_applyJT_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_applyJT1_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (CudaVec3<float>*)out, (const float*)in); mycudaDebugError("SubsetMappingCuda3f_applyJT1_kernel<CudaVec3<float> >");}
}


void SubsetMappingCuda3f1_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f1_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_applyJT_kernel<CudaVec4<float> ><<< grid, threads >>>(insize, maxNOut, (const int*)mapT, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_applyJT_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_applyJT1_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_applyJT1_kernel<CudaVec4<float> >");}
}


void SubsetMappingCuda3f1_3f_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f_apply_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f1_3f_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f_apply_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_applyJT_kernel<CudaVec3<float> ><<< grid, threads >>>(insize, maxNOut, (const int*)mapT, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_applyJT_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f1_3f_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_applyJT1_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (const int*)map, (CudaVec4<float>*)out, (const float*)in); mycudaDebugError("SubsetMappingCuda3f_applyJT1_kernel<CudaVec4<float> >");}
}


void SubsetMappingCuda3f_3f1_apply(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f_3f1_applyJ(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void SubsetMappingCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f_applyJT_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const int*)mapT, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f_applyJT_kernel<CudaVec4<float> >");}
}

void SubsetMappingCuda3f_3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {SubsetMappingCuda3f1_applyJT1_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int*)map, (CudaVec3<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SubsetMappingCuda3f1_applyJT1_kernel<CudaVec3<float> >");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
