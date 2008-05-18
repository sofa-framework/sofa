#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

#if defined(__cplusplus) && CUDA_VERSION != 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void FixedConstraintCuda1t_projectResponseContiguous_kernel(int size, real* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = 0.0f;
}

template<class real>
__global__ void FixedConstraintCuda3t_projectResponseContiguous_kernel(int size, CudaVec3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda3t1_projectResponseContiguous_kernel(int size, CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[index] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda1t_projectResponseIndexed_kernel(int size, const int* indices, real* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = 0.0f;
}

template<class real>
__global__ void FixedConstraintCuda3t_projectResponseIndexed_kernel(int size, const int* indices, CudaVec3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

template<class real>
__global__ void FixedConstraintCuda3t1_projectResponseIndexed_kernel(int size, const int* indices, CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
        dx[indices[index]] = CudaVec4<real>::make(0.0f,0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t_projectResponseContiguous_kernel<float><<< grid, threads >>>(size, (CudaVec3<float>*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<float><<< grid, threads >>>(3*size, (float*)dx);
    cudaMemset(dx, 0, size*3*sizeof(float));
}

void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t1_projectResponseContiguous_kernel<float><<< grid, threads >>>(size, (CudaVec4<float>*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<float><<< grid, threads >>>(4*size, (float*)dx);
    cudaMemset(dx, 0, size*4*sizeof(float));
}

void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<float>*)dx);
}

void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3t1_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<float>*)dx);
}

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t_projectResponseContiguous_kernel<double><<< grid, threads >>>(size, (CudaVec3<double>*)dx);
    //dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<double><<< grid, threads >>>(3*size, (double*)dx);
    cudaMemset(dx, 0, size*3*sizeof(double));
}

void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda3t1_projectResponseContiguous_kernel<double><<< grid, threads >>>(size, (CudaVec4<double>*)dx);
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //FixedConstraintCuda1t_projectResponseContiguous_kernel<double><<< grid, threads >>>(4*size, (double*)dx);
    cudaMemset(dx, 0, size*4*sizeof(double));
}

void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<double>*)dx);
}

void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    FixedConstraintCuda3t1_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec4<double>*)dx);
}

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#if defined(__cplusplus) && CUDA_VERSION != 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
