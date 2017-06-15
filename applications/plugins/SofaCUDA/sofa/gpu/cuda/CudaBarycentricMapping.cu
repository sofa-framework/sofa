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
    void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f_3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f1_3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void SparseGridMapperCuda3f_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f1_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f_3f1_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_3f1_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f1_3f_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_3f_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);


    void MeshMapperCuda3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f_3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);

    void MeshMapperCuda3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f_3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
}

struct __align__(16) GPUCubeData
{
    int i;
    float fx,fy,fz;
};

struct __align__(8) GPULinearMap
{
    int i;
    float f;
};

#define SPARSE_GRID_BSIZE 256
#define SPARSE_GRID_MAX_CUBE 32

//////////////////////
// GPU-side methods //
//////////////////////

template<class TIn>
__global__ void RegularGridMapperCuda3f_apply_kernel(unsigned int size, unsigned int nx, unsigned int nxny, const GPUCubeData* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);

    GPUCubeData c = map[index0+index1];
    if (index0+index1 < size)
    {
        //const Real fx = map[i].baryCoords[0];
        //const Real fy = map[i].baryCoords[1];
        //const Real fz = map[i].baryCoords[2];
        //out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
        //       + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
        //       + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
        //       + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
        //       + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
        //       + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
        //       + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
        //       + in[cube[7]] * ((  fx) * (  fy) * (  fz));

        res = CudaVec3<float>::make(in [c.i          ]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i+1        ]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i  +nx     ]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i+1+nx     ]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i     +nxny]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i+1   +nxny]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i  +nx+nxny]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i+1+nx+nxny]) * ((  c.fx) * (  c.fy) * (  c.fz));
    }

    //__syncthreads();

    const int index3 = umul24(3,index1);

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += umul24(index0,3);
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}


template<class TIn>
__global__ void RegularGridMapperCuda3f1_apply_kernel(unsigned int size, unsigned int nx, unsigned int nxny, const GPUCubeData* map, CudaVec4<float>* out, const TIn* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);

    GPUCubeData c = map[index];
    if (index < size)
    {
        //const Real fx = map[i].baryCoords[0];
        //const Real fy = map[i].baryCoords[1];
        //const Real fz = map[i].baryCoords[2];
        //out[i] = in[cube[0]] * ((1-fx) * (1-fy) * (1-fz))
        //       + in[cube[1]] * ((  fx) * (1-fy) * (1-fz))
        //       + in[cube[2]] * ((1-fx) * (  fy) * (1-fz))
        //       + in[cube[3]] * ((  fx) * (  fy) * (1-fz))
        //       + in[cube[4]] * ((1-fx) * (1-fy) * (  fz))
        //       + in[cube[5]] * ((  fx) * (1-fy) * (  fz))
        //       + in[cube[6]] * ((1-fx) * (  fy) * (  fz))
        //       + in[cube[7]] * ((  fx) * (  fy) * (  fz));

        res = CudaVec3<float>::make(in [c.i          ]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i+1        ]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i  +nx     ]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i+1+nx     ]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [c.i     +nxny]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i+1   +nxny]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i  +nx+nxny]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [c.i+1+nx+nxny]) * ((  c.fx) * (  c.fy) * (  c.fz));
    }

    out[index] = CudaVec4<float>::make(res);
}

template<class TIn>
__global__ void RegularGridMapperCuda3f_applyJT_kernel(unsigned int size, unsigned int maxNOut, const GPULinearMap* mapT, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        GPULinearMap data = *mapT;
        mapT+=BSIZE;
        if (data.i != -1)
            res += CudaVec3<float>::make(in[data.i]) * data.f;
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

template<class TIn>
__global__ void RegularGridMapperCuda3f1_applyJT_kernel(unsigned int size, unsigned int maxNOut, const GPULinearMap* mapT, CudaVec4<float>* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0+index1;

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        GPULinearMap data = *mapT;
        mapT+=BSIZE;
        if (data.i != -1)
            res += CudaVec3<float>::make(in [data.i]) * data.f;
    }

    CudaVec4<float> o = out[index];
    o.x += res.x;
    o.y += res.y;
    o.z += res.z;
    out[index] = o;
}





template<class TIn>
__global__ void MeshMapperCuda3f_apply_kernel(unsigned int size, unsigned int maxN, const GPULinearMap* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    map+=umul24(index0,maxN)+index1;
    for (int s = 0; s < maxN; s++)
    {
        GPULinearMap data = *map;
        map+=BSIZE;
        if (data.i != 0)
            res += CudaVec3<float>::make(in[data.i-1]) * data.f;
    }

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

template<class TIn>
__global__ void MeshMapperCuda3f1_apply_kernel(unsigned int size, unsigned int maxN, const GPULinearMap* map, CudaVec4<float>* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0+index1;

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    map+=umul24(index0,maxN)+index1;
    for (int s = 0; s < maxN; s++)
    {
        GPULinearMap data = *map;
        map+=BSIZE;
        if (data.i != 0)
            res += CudaVec3<float>::make(in[data.i-1]) * data.f;
    }

    CudaVec4<float> o;
    o.x = res.x;
    o.y = res.y;
    o.z = res.z;
    o.w = 0.0f;
    out[index] = o;
}






template<class TIn>
__global__ void MeshMapperCuda3f_applyPEq_kernel(unsigned int size, unsigned int maxN, const GPULinearMap* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    map+=umul24(index0,maxN)+index1;
    for (int s = 0; s < maxN; s++)
    {
        GPULinearMap data = *map;
        map+=BSIZE;
        if (data.i != 0)
            res += CudaVec3<float>::make(in[data.i-1]) * data.f;
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

template<class TIn>
__global__ void MeshMapperCuda3f1_applyPEq_kernel(unsigned int size, unsigned int maxN, const GPULinearMap* map, CudaVec4<float>* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0+index1;

    CudaVec3<float> res = CudaVec3<float>::make(0,0,0);
    //res += *in * mapT[index0+index1].f;

    map+=umul24(index0,maxN)+index1;
    for (int s = 0; s < maxN; s++)
    {
        GPULinearMap data = *map;
        map+=BSIZE;
        if (data.i != 0)
            res += CudaVec3<float>::make(in[data.i-1]) * data.f;
    }

    CudaVec4<float> o = out[index];
    o.x += res.x;
    o.y += res.y;
    o.z += res.z;
    out[index] = o;
}

template<class TIn>
__global__ void SparseGridMapperCuda3f_apply_kernel(unsigned int size, const int * cudaHexa, const GPUCubeData* map, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,SPARSE_GRID_BSIZE);
    const int index1 = threadIdx.x;
    const int index3 = umul24(3,index1);
    out += umul24(index0,3);

    if (index0+index1 < size)
    {
        GPUCubeData c = map[index0+index1];
        const int * cube = cudaHexa + c.i*8;
        CudaVec3<float> res = CudaVec3<float>::make(in [cube[0]]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[1]]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[3]]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[2]]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[4]]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[5]]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[7]]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[6]]) * ((  c.fx) * (  c.fy) * (  c.fz));

        out[index3  ] = res.x;
        out[index3+1] = res.y;
        out[index3+2] = res.z;
    }
}

template<class TIn>
__global__ void SparseGridMapperCuda3f1_apply_kernel(unsigned int size, const int * cudaHexa, const GPUCubeData* map, CudaVec4<float>* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,SPARSE_GRID_BSIZE);
    const int index1 = threadIdx.x;

    if (index0+index1 < size)
    {
        GPUCubeData c = map[index0+index1];
        const int * cube = cudaHexa + c.i*8;
        CudaVec3<float> res = CudaVec3<float>::make(in [cube[0]]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[1]]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[3]]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[2]]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + CudaVec3<float>::make(in [cube[4]]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[5]]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[7]]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + CudaVec3<float>::make(in [cube[6]]) * ((  c.fx) * (  c.fy) * (  c.fz));
        out[index0+index1] = CudaVec4<float>::make(res);
    }
}

template<class TIn>
__global__ void SparseGridMapperCuda3f_applyJT_kernel(unsigned int size, const unsigned * CudaTnb, const unsigned * CudaTst, const unsigned * CudaTid, const float * CudaTVal, float* out, const TIn* in)
{
    const int bi = blockIdx.x;
    const int tx = threadIdx.x;

    __shared__ CudaVec3<float> readValue[SPARSE_GRID_MAX_CUBE];
    __shared__ unsigned sto;
    __shared__ unsigned nbo;

    if (!tx)
    {
        sto = CudaTst[bi];
        nbo = CudaTnb[bi];
    }

    __syncthreads();

    readValue[tx] = CudaVec3<float>::make(0,0,0);

    for (unsigned nx=sto+tx; nx<sto+nbo; nx+=SPARSE_GRID_MAX_CUBE)
    {
        readValue[tx] += CudaVec3<float>::make(in[CudaTid[nx]]) * CudaTVal[nx];
    }

#if SPARSE_GRID_MAX_CUBE>128
    __syncthreads();
    if (tx<128) readValue[tx] += readValue[tx+128];
#endif
#if SPARSE_GRID_MAX_CUBE>64
    __syncthreads();
    if (tx<64) readValue[tx] += readValue[tx+64];
#endif
#if SPARSE_GRID_MAX_CUBE>32
    __syncthreads();
    if (tx<32) readValue[tx] += readValue[tx+32];
#endif
#if SPARSE_GRID_MAX_CUBE>16
    if (tx<16)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+16)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+16)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+16)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>8
    if (tx<8)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+8)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+8)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+8)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>4
    if (tx<4)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+4)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+4)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+4)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>2
    if (tx<2)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+2)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+2)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+2)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>1
    if (tx<1)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+1)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+1)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+1)[2];
    }
#endif

    if (tx<3)
    {
        out += umul24(bi,3);
        out[tx] += ((float*) &readValue[0])[tx];
    }
}

template<class TIn>
__global__ void SparseGridMapperCuda3f1_applyJT_kernel(unsigned int size, const unsigned * CudaTnb, const unsigned * CudaTst, const unsigned * CudaTid, const float * CudaTVal, CudaVec4<float>* out, const TIn* in)
{
    const int bi = blockIdx.x;
    const int tx = threadIdx.x;

    __shared__ CudaVec3<float> readValue[SPARSE_GRID_MAX_CUBE];
    __shared__ unsigned sto;
    __shared__ unsigned nbo;

    if (!tx)
    {
        sto = CudaTst[bi];
        nbo = CudaTnb[bi];
    }

    __syncthreads();

    readValue[tx] = CudaVec3<float>::make(0,0,0);

    for (unsigned nx=sto+tx; nx<sto+nbo; nx+=SPARSE_GRID_MAX_CUBE)
    {
        readValue[tx] += CudaVec3<float>::make(in[CudaTid[nx]]) * CudaTVal[nx];
    }

#if SPARSE_GRID_MAX_CUBE>128
    __syncthreads();
    if (tx<128) readValue[tx] += readValue[tx+128];
#endif
#if SPARSE_GRID_MAX_CUBE>64
    __syncthreads();
    if (tx<64) readValue[tx] += readValue[tx+64];
#endif
#if SPARSE_GRID_MAX_CUBE>32
    __syncthreads();
    if (tx<32) readValue[tx] += readValue[tx+32];
#endif
#if SPARSE_GRID_MAX_CUBE>16
    if (tx<16)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+16)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+16)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+16)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>8
    if (tx<8)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+8)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+8)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+8)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>4
    if (tx<4)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+4)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+4)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+4)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>2
    if (tx<2)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+2)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+2)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+2)[2];
    }
#endif
#if SPARSE_GRID_MAX_CUBE>1
    if (tx<1)
    {
        ((volatile float *) readValue+tx)[0] += ((volatile float *) readValue+tx+1)[0];
        ((volatile float *) readValue+tx)[1] += ((volatile float *) readValue+tx+1)[1];
        ((volatile float *) readValue+tx)[2] += ((volatile float *) readValue+tx+1)[2];
    }
#endif


    if (tx==0)
    {
        CudaVec4<float> o = out[bi];
        o.x += readValue[0].x;
        o.y += readValue[0].y;
        o.z += readValue[0].z;
        out[bi] = o;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_apply_kernel<CudaVec3<float> >");}
}

void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_apply_kernel<CudaVec3<float> >");}
}

void RegularGridMapperCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_applyJT_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_applyJT_kernel<CudaVec3<float> >");}
}

void RegularGridMapperCuda3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void RegularGridMapperCuda3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void RegularGridMapperCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_applyJT_kernel<CudaVec4<float> ><<< grid, threads >>>(insize, maxNOut, (const GPULinearMap*)mapT, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_applyJT_kernel<CudaVec4<float> >");}
}


void RegularGridMapperCuda3f1_3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_apply_kernel<CudaVec4<float> >");}
}

void RegularGridMapperCuda3f1_3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_apply_kernel<CudaVec4<float> >");}
}

void RegularGridMapperCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f_applyJT_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("RegularGridMapperCuda3f_applyJT_kernel<CudaVec4<float> >");}
}


void RegularGridMapperCuda3f_3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void RegularGridMapperCuda3f_3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void RegularGridMapperCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    {RegularGridMapperCuda3f1_applyJT_kernel<CudaVec3<float> ><<< grid, threads >>>(insize, maxNOut, (const GPULinearMap*)mapT, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("RegularGridMapperCuda3f1_applyJT_kernel<CudaVec3<float> >");}
}





void SparseGridMapperCuda3f_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_apply_kernel<CudaVec3<float> >");}
}

void SparseGridMapperCuda3f_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_apply_kernel<CudaVec3<float> >");}
}

void SparseGridMapperCuda3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_MAX_CUBE,1);
    dim3 grid(size,1);
    {SparseGridMapperCuda3f_applyJT_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const unsigned *) CudaTnb, (const unsigned *) CudaTst, (const unsigned *) CudaTid, (const float *) CudaTVal, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_applyJT_kernel<CudaVec3<float> >");}
}

void SparseGridMapperCuda3f1_apply(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void SparseGridMapperCuda3f1_applyJ(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void SparseGridMapperCuda3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid(size,1);
    {SparseGridMapperCuda3f1_applyJT_kernel<CudaVec4<float> ><<< grid, threads >>>(size,  (const unsigned *) CudaTnb, (const unsigned *) CudaTst, (const unsigned *) CudaTid, (const float *) CudaTVal, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_applyJT_kernel<CudaVec4<float> >");}
}


void SparseGridMapperCuda3f1_3f_apply(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_apply_kernel<CudaVec4<float> >");}
}

void SparseGridMapperCuda3f1_3f_applyJ(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_apply_kernel<CudaVec4<float> >");}
}

void SparseGridMapperCuda3f1_3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_MAX_CUBE,1);
    dim3 grid(size,1);
    {SparseGridMapperCuda3f_applyJT_kernel<CudaVec4<float> ><<< grid, threads >>>(size,  (const unsigned *) CudaTnb, (const unsigned *) CudaTst, (const unsigned *) CudaTid, (const float *) CudaTVal, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("SparseGridMapperCuda3f_applyJT_kernel<CudaVec4<float> >");}
}


void SparseGridMapperCuda3f_3f1_apply(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void SparseGridMapperCuda3f_3f1_applyJ(unsigned int size, const void * cudaHexa, const void* map, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_BSIZE,1);
    dim3 grid((size+SPARSE_GRID_BSIZE-1)/SPARSE_GRID_BSIZE,1);
    {SparseGridMapperCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int *) cudaHexa, (const GPUCubeData*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void SparseGridMapperCuda3f_3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in)
{
    dim3 threads(SPARSE_GRID_MAX_CUBE,1);
    dim3 grid(size,1);
    {SparseGridMapperCuda3f1_applyJT_kernel<CudaVec3<float> ><<< grid, threads >>>(size,  (const unsigned *) CudaTnb, (const unsigned *) CudaTst, (const unsigned *) CudaTid, (const float *) CudaTVal, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("SparseGridMapperCuda3f1_applyJT_kernel<CudaVec3<float> >");}
}










void MeshMapperCuda3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f_apply_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, maxN, (const GPULinearMap*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("MeshMapperCuda3f_apply_kernel<CudaVec3<float> >");}
}

void MeshMapperCuda3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f1_apply_kernel<CudaVec4<float> ><<< grid, threads >>>(size, maxN, (const GPULinearMap*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("MeshMapperCuda3f1_apply_kernel<CudaVec4<float> >");}
}

void MeshMapperCuda3f_3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f1_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, maxN, (const GPULinearMap*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("MeshMapperCuda3f1_apply_kernel<CudaVec3<float> >");}
}

void MeshMapperCuda3f1_3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f_apply_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, maxN, (const GPULinearMap*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("MeshMapperCuda3f_apply_kernel<CudaVec4<float> >");}
}


void MeshMapperCuda3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f_applyPEq_kernel<CudaVec3<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, maxN, (const GPULinearMap*)map, (float*)out, (const CudaVec3<float>*)in); mycudaDebugError("MeshMapperCuda3f_applyPEq_kernel<CudaVec3<float> >");}
}

void MeshMapperCuda3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f1_applyPEq_kernel<CudaVec4<float> ><<< grid, threads >>>(size, maxN, (const GPULinearMap*)map, (CudaVec4<float>*)out, (const CudaVec4<float>*)in); mycudaDebugError("MeshMapperCuda3f1_applyPEq_kernel<CudaVec4<float> >");}
}

void MeshMapperCuda3f_3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f1_applyPEq_kernel<CudaVec3<float> ><<< grid, threads >>>(size, maxN, (const GPULinearMap*)map, (CudaVec4<float>*)out, (const CudaVec3<float>*)in); mycudaDebugError("MeshMapperCuda3f1_applyPEq_kernel<CudaVec3<float> >");}
}

void MeshMapperCuda3f1_3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMapperCuda3f_applyPEq_kernel<CudaVec4<float> ><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, maxN, (const GPULinearMap*)map, (float*)out, (const CudaVec4<float>*)in); mycudaDebugError("MeshMapperCuda3f_applyPEq_kernel<CudaVec4<float> >");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
