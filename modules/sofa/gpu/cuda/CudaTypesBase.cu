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
#include "CudaCommon.h"
#include "CudaMath.h"
#include "mycuda.h"
#include <cuda.h>

#if defined(__cplusplus) && CUDA_VERSION < 2000
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

using namespace sofa::gpu::cuda;

extern "C"
{
    void matrix_vector_productf(int dim,const void * M,int mPitch,const void * r,void * z);
    void matrix_vector_productd(int dim,const void * M,int mPitch,const void * r,void * z);
}

#define MAX_THREADS 256

//Solve z = M^-1 * r
//for (unsigned j=0;j<n;j++) {
//	CudaT[j] = 0.0;
//	for (unsigned i=0;i<n;i++) {
//		//initial matrix + current rotations
//		CudaT[j] += M[j][i] * z[i];
//	}
//}
template<class real>
__global__ void matrix_vector_product_kernel(int dim,const real * M,int mPitch,const real * r, real * z,int offset)
{
    __shared__ real stemp[MAX_THREADS];
    volatile real * temp = stemp;

    int tx = threadIdx.x;
    int by = blockIdx.y;
    real * m_i = ((real *) (((char*) M) + (mPitch*by)));

    temp[tx] = 0.0;
    for(int i=tx; i<dim; i+=MAX_THREADS) temp[tx] += m_i[i] * r[i];

#if MAX_THREADS > 128
    __syncthreads();
    if (tx < 128) temp[tx]+=temp[tx+128];
#endif
#if MAX_THREADS > 64
    __syncthreads();
    if (tx < 64) temp[tx]+=temp[tx+64];
#endif
#if MAX_THREADS > 32
    __syncthreads();
    if (tx < 32) temp[tx]+=temp[tx+32];
#endif
#if MAX_THREADS > 16
    if (tx < 16) temp[tx]+=temp[tx+16];
#endif
#if MAX_THREADS > 8
    if (tx < 8) temp[tx]+=temp[tx+8];
#endif
#if MAX_THREADS > 4
    if (tx < 4) temp[tx]+=temp[tx+4];
#endif
#if MAX_THREADS > 2
    if (tx < 2) temp[tx]+=temp[tx+2];
#endif
#if MAX_THREADS > 1
    if (tx < 1) temp[tx]+=temp[tx+1];
#endif
    if (! tx) z[by] = temp[0];
}

void matrix_vector_productf(int n,const void * M,int mPitch,const void * r,void * z)
{
    dim3 threads(MAX_THREADS,1);
    dim3 grid(1,n);

    {matrix_vector_product_kernel<float><<< grid, threads >>>(n,(const float*)M, mPitch,(const float*)r, (float*)z,MAX_THREADS/2); mycudaDebugError("matrix_vector_product_kernel<float>");}
}

void matrix_vector_productd(int n,const void * M,int mPitch,const void * r,void * z)
{
    dim3 threads(MAX_THREADS,1);
    dim3 grid(1,n);

    {matrix_vector_product_kernel<double><<< grid, threads >>>(n,(const double*)M, mPitch,(const double*)r, (double*)z,MAX_THREADS/2); mycudaDebugError("matrix_vector_product_kernel<double>");}
}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
