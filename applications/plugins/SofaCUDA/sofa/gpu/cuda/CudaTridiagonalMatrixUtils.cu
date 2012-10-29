/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/gpu/cuda/CudaCommon.h>
#include <sofa/gpu/cuda/CudaMath.h>
#include <sofa/gpu/cuda/mycuda.h>
#include <cuda.h>

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa {
namespace gpu {
namespace cuda {
#endif

using namespace sofa::gpu::cuda;

extern "C"
{
	void trimatrix_vector_productf(int dim,const void * R,const void * r,void * z);
#ifdef SOFA_GPU_CUDA_DOUBLE
	void trimatrix_vector_productd(int dim,const void * R,const void * r,void * z);
#endif

	void trimatrixtr_vector_productf(int dim,const void * R,const void * r,void * z);
#ifdef SOFA_GPU_CUDA_DOUBLE
	void trimatrixtr_vector_productd(int dim,const void * R,const void * r,void * z);
#endif

    void trimatrix_trimatrixtr_productf(int dim3,const void * R1,const void * R2,void * Rout);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void trimatrix_trimatrixtr_productd(int dim3,const void * R1,const void * R2,void * Rout);
#endif
}

//have to be multiple of 3
#define NB_THREAD_TR 96

//Solve z = R^t * b
//k = 0;
//l = 0;
//while (l < n) {
//	z[l+0] = CudaR[k + 0] * r[l + 0] + CudaR[k + 3] * r[l + 1] + CudaR[k + 6] * r[l + 2];
//	z[l+1] = CudaR[k + 1] * r[l + 0] + CudaR[k + 4] * r[l + 1] + CudaR[k + 7] * r[l + 2];
//	z[l+2] = CudaR[k + 2] * r[l + 0] + CudaR[k + 5] * r[l + 1] + CudaR[k + 8] * r[l + 2];
//	l+=3;
//	k+=9;
//}
template<class real>
__global__ void trimatrix_vector_product_kernel(int dim,const real * R,const real * b, real * z) {
	__shared__ volatile real sb[NB_THREAD_TR];
	__shared__ volatile real sr[NB_THREAD_TR*3];
	
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int wadd = bx*NB_THREAD_TR;
	int wadd3 = wadd*3;
	
    sr[tx                          ] = R[wadd3+tx                          ];
    sr[tx+NB_THREAD_TR             ] = R[wadd3+tx+NB_THREAD_TR             ];
    sr[tx+NB_THREAD_TR+NB_THREAD_TR] = R[wadd3+tx+NB_THREAD_TR+NB_THREAD_TR];

    if (wadd+tx>=dim) return;

    sb[tx] = b[wadd+tx];

	__syncthreads();

    int txiv3 = (tx/3)*3;
    int tx3 = tx*3;
    z[wadd+tx] = sb[txiv3] * sr[tx3] + sb[txiv3+1] * sr[tx3+1] + sb[txiv3+2] * sr[tx3+2];
}

void trimatrix_vector_productf(int n,const void * R,const void * r,void * z) {
	dim3 threads(NB_THREAD_TR,1);
	dim3 grid((n+NB_THREAD_TR-1)/NB_THREAD_TR,1);

    {trimatrix_vector_product_kernel<float><<< grid, threads >>>(n,(const float*)R,(const float*)r, (float*)z); mycudaDebugError("trimatrix_vector_product_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void trimatrix_vector_productd(int n,const void * R,const void * r,void * z) {

	dim3 threads(NB_THREAD_TR,1);
	dim3 grid((n+NB_THREAD_TR-1)/NB_THREAD_TR,1);

    {trimatrix_vector_product_kernel<double><<< grid, threads >>>(n,(const double*)R,(const double*)r, (double*)z); mycudaDebugError("trimatrix_vector_product_kernel<double>");}

}
#endif

//Solve z = R * tmp
//k = 0;
//l = 0;
//while (l < n) {
//	z[l+0] = CudaR[k + 0] * CudaT[l + 0] + CudaR[k + 1] * CudaT[l + 1] + CudaR[k + 2] * CudaT[l + 2];
//	z[l+1] = CudaR[k + 3] * CudaT[l + 0] + CudaR[k + 4] * CudaT[l + 1] + CudaR[k + 5] * CudaT[l + 2];
//	z[l+2] = CudaR[k + 6] * CudaT[l + 0] + CudaR[k + 7] * CudaT[l + 1] + CudaR[k + 8] * CudaT[l + 2];
//	l+=3;
//	k+=9;
//}

template<class real>
__global__ void trimatrixtr_vector_product_kernel(int dim, int dim3,const real * R,const real * b, real * z) {
    __shared__ volatile real sb[NB_THREAD_TR];
    __shared__ volatile real sr[NB_THREAD_TR*3];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int wadd = bx*NB_THREAD_TR;
    int wadd3 = wadd*3;

    sr[tx                          ] = R[wadd3+tx                          ];
    sr[tx+NB_THREAD_TR             ] = R[wadd3+tx+NB_THREAD_TR             ];
    sr[tx+NB_THREAD_TR+NB_THREAD_TR] = R[wadd3+tx+NB_THREAD_TR+NB_THREAD_TR];

    if (wadd+tx>=dim) return;

    sb[tx] = b[wadd+tx];

    __syncthreads();

    int txiv3 = (tx/3)*3;
    int tx3 = (tx/3)*9 + (tx%3);
    z[wadd+tx] = sb[txiv3] * sr[tx3] + sb[txiv3+1] * sr[tx3+3] + sb[txiv3+2] * sr[tx3+6];
}

void trimatrixtr_vector_productf(int n,const void * R,const void * r,void * z) {
    dim3 threads(NB_THREAD_TR,1);
    dim3 grid((n+NB_THREAD_TR-1)/NB_THREAD_TR,1);

    {trimatrixtr_vector_product_kernel<float><<< grid, threads >>>(n,n*3,(const float*)R,(const float*)r, (float*)z); mycudaDebugError("trimatrixtr_vector_product_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void trimatrixtr_vector_productd(int n,const void * R,const void * r,void * z) {
    dim3 threads(NB_THREAD_TR,1);
    dim3 grid((n+NB_THREAD_TR-1)/NB_THREAD_TR,1);

    {trimatrixtr_vector_product_kernel<double><<< grid, threads >>>(n,n*3,(const double*)R,(const double*)r, (double*)z); mycudaDebugError("trimatrixtr_vector_product_kernel<double>");}
}
#endif

#define NB_THREAD_TRTR 288

// Rout = R1 * tr(R2)
template<class real>
__global__ void trimatrix_trimatrixtr_product_kernel(int dim3,const real * R1,const real * R2, real * Rout) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int wad = bx*NB_THREAD_TRTR + tx;
	real tmp;
	
    if (wad>=dim3) return;

    int tx9 = wad % 9;
    int wa3 = wad - tx9;
    int idR1 = (tx9 / 3) * 3;
    int idR2 = (tx9 % 3) * 3;

    tmp = R1[wa3 + idR1 + 0] * R2[wa3 + idR2 + 0] +
          R1[wa3 + idR1 + 1] * R2[wa3 + idR2 + 1] +
          R1[wa3 + idR1 + 2] * R2[wa3 + idR2 + 2];

	__syncthreads(); // syncthreads and tmp variable to allow rinv == rout or rcur == rout

	if (wad<dim3) Rout[wad] = tmp;
}

void trimatrix_trimatrixtr_productf(int dimp3,const void * R1,const void * R2,void * Rout) {
	dim3 threads(NB_THREAD_TRTR,1);
	dim3 grid((dimp3+NB_THREAD_TRTR-1)/NB_THREAD_TRTR,1);

    {trimatrix_trimatrixtr_product_kernel<float><<< grid, threads >>>(dimp3,(const float*)R1,(const float*)R2, (float*)Rout); mycudaDebugError("trimatrix_trimatrixtr_product_kernel<float>");}
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void trimatrix_trimatrixtr_productd(int dimp3,const void * R1,const void * R2,void * Rout) {
	dim3 threads(NB_THREAD_TRTR,1);
	dim3 grid((dimp3+NB_THREAD_TRTR-1)/NB_THREAD_TRTR,1);

    {trimatrix_trimatrixtr_product_kernel<double><<< grid, threads >>>(dimp3,(const double*)R1,(const double*)R2, (double*)Rout); mycudaDebugError("trimatrix_trimatrixtr_product_kernel<double>");}
}
#endif

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif


//#endif
