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
    void RegularGridMapperCuda3f_applyJT(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
}

struct GPUCubeData
{
    int i;
    float fx,fy,fz;
};

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void RegularGridMapperCuda3f_apply_kernel(unsigned int size, unsigned int nx, unsigned int nxny, const GPUCubeData* map, float* out, const float* in)
{
    int index0 = blockIdx.x*BSIZE; //blockDim.x;
    int index1 = threadIdx.x;

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

        res = ((const float3*)in) [c.i          ] * ((1-c.fx) * (1-c.fy) * (1-c.fz))
                + ((const float3*)in) [c.i+1        ] * ((  c.fx) * (1-c.fy) * (1-c.fz))
                + ((const float3*)in) [c.i  +nx     ] * ((1-c.fx) * (  c.fy) * (1-c.fz))
                + ((const float3*)in) [c.i+1+nx     ] * ((  c.fx) * (  c.fy) * (1-c.fz))
                + ((const float3*)in) [c.i     +nxny] * ((1-c.fx) * (1-c.fy) * (  c.fz))
                + ((const float3*)in) [c.i+1   +nxny] * ((  c.fx) * (1-c.fy) * (  c.fz))
                + ((const float3*)in) [c.i  +nx+nxny] * ((1-c.fx) * (  c.fy) * (  c.fz))
                + ((const float3*)in) [c.i+1+nx+nxny] * ((  c.fx) * (  c.fy) * (  c.fz));
    }

    //__syncthreads();

    int index3 = 3*index1;

    temp[index3  ] = res.x;
    temp[index3+1] = res.y;
    temp[index3+2] = res.z;

    __syncthreads();

    out += index0*3;
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float*)in);
}

void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    RegularGridMapperCuda3f_apply_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float*)in);
}

void RegularGridMapperCuda3f_applyJT(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    //RegularGridMapperCuda3f_applyJT_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, gridsize[0], gridsize[0]*gridsize[1], (const GPUCubeData*)map, (float*)out, (const float*)in);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
