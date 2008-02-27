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

    float3 res = make_float3(0,0,0);

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

        res = make_float3(in [c.i          ]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + make_float3(in [c.i+1        ]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + make_float3(in [c.i  +nx     ]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + make_float3(in [c.i+1+nx     ]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + make_float3(in [c.i     +nxny]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + make_float3(in [c.i+1   +nxny]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + make_float3(in [c.i  +nx+nxny]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + make_float3(in [c.i+1+nx+nxny]) * ((  c.fx) * (  c.fy) * (  c.fz));
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
__global__ void RegularGridMapperCuda3f1_apply_kernel(unsigned int size, unsigned int nx, unsigned int nxny, const GPUCubeData* map, float4* out, const TIn* in)
{
    const int index = umul24(blockIdx.x,BSIZE) + threadIdx.x;

    float3 res = make_float3(0,0,0);

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

        res = make_float3(in [c.i          ]) * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + make_float3(in [c.i+1        ]) * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + make_float3(in [c.i  +nx     ]) * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + make_float3(in [c.i+1+nx     ]) * ((  c.fx) * (  c.fy) * (1-c.fz))
                + make_float3(in [c.i     +nxny]) * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + make_float3(in [c.i+1   +nxny]) * ((  c.fx) * (1-c.fy) * (  c.fz))
                + make_float3(in [c.i  +nx+nxny]) * ((1-c.fx) * (  c.fy) * (  c.fz))
                + make_float3(in [c.i+1+nx+nxny]) * ((  c.fx) * (  c.fy) * (  c.fz));
    }

    out[index] = make_float4(res);
}

template<class TIn>
__global__ void RegularGridMapperCuda3f_applyJT_kernel(unsigned int size, unsigned int maxNOut, const GPULinearMap* mapT, float* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    float3 res = make_float3(0,0,0);
    //res += *in * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        GPULinearMap data = *mapT;
        mapT+=BSIZE;
        if (data.i != -1)
            res += make_float3(in[data.i]) * data.f;
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
__global__ void RegularGridMapperCuda3f1_applyJT_kernel(unsigned int size, unsigned int maxNOut, const GPULinearMap* mapT, float4* out, const TIn* in)
{
    const int index0 = umul24(blockIdx.x,BSIZE); //blockDim.x;
    const int index1 = threadIdx.x;
    const int index = index0+index1;

    float3 res = make_float3(0,0,0);
    //res += *in * mapT[index0+index1].f;

    mapT+=umul24(index0,maxNOut)+index1;
    for (int s = 0; s < maxNOut; s++)
    {
        GPULinearMap data = *mapT;
        mapT+=BSIZE;
        if (data.i != -1)
            res += make_float3(in [data.i]) * data.f;
    }

    float4 o = out[index];
    o.x += res.x;
    o.y += res.y;
    o.z += res.z;
    out[index] = o;
}

//////////////////////
// CPU-side methods //
//////////////////////

void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float3*)in);
}

void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float3*)in);
}

void RegularGridMapperCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_applyJT_kernel<float3><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float*)out, (const float3*)in);
}


void RegularGridMapperCuda3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_apply_kernel<float4><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float4*)out, (const float4*)in);
}

void RegularGridMapperCuda3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_apply_kernel<float4><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float4*)out, (const float4*)in);
}

void RegularGridMapperCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_applyJT_kernel<float4><<< grid, threads >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float4*)out, (const float4*)in);
}


void RegularGridMapperCuda3f1_3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float4*)in);
}

void RegularGridMapperCuda3f1_3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float4*)in);
}

void RegularGridMapperCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_applyJT_kernel<float4><<< grid, threads, BSIZE*3*sizeof(float) >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float*)out, (const float4*)in);
}


void RegularGridMapperCuda3f_3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_apply_kernel<float3><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float4*)out, (const float3*)in);
}

void RegularGridMapperCuda3f_3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_apply_kernel<float3><<< grid, threads >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float4*)out, (const float3*)in);
}

void RegularGridMapperCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((insize+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f1_applyJT_kernel<float3><<< grid, threads >>>(insize, maxNOut, (const GPULinearMap*)mapT, (float4*)out, (const float3*)in);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
