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
    void ExternalForceFieldCuda3f_addForce(unsigned int size,void* f, const void* indices,const void *forces);
}

//////////////////////
// GPU-side methods //
//////////////////////

__global__ void ExternalForceFieldCuda3f_addForce_kernel(int size,float * f, const unsigned * indices,const float *forces)
{
    int index0 = blockIdx.x*BSIZE;
    int index0_3 = index0*3;

    forces += index0_3;
    indices += index0_3;
    f += index0_3;

    int index = threadIdx.x;
    int index_3 = index*3;

    //! Dynamically allocated shared memory to reorder global memory access
    extern  __shared__  float temp[];

    if((index0+index)<size)
    {
        f[indices[index_3]  ] += forces[index_3];
        f[indices[index_3]+1] += forces[index_3+1];
        f[indices[index_3]+2] += forces[index_3+2];
    }
}


//////////////////////
// CPU-side methods //
//////////////////////


void ExternalForceFieldCuda3f_addForce(unsigned int size,void* f, const void* indices,const void *forces)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    ExternalForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (float*)f, (const unsigned*)indices,(const float*)forces);
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
