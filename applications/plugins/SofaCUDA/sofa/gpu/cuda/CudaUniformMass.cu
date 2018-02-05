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
#include "CudaMathRigid.h"
#include "cuda.h"

#include <cstdio>

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
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);

    void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f);
	
	void UniformMassCudaRigid3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
	void UniformMassCudaRigid3f_accFromF(unsigned int size, float mass, void* a, const void* dx);
	void UniformMassCudaRigid3f_addForce(unsigned int size, const float* mg, void* f);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void UniformMassCuda3d_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d_addForce(unsigned int size, const double *mg, void* f);

    void UniformMassCuda3d1_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d1_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d1_addForce(unsigned int size, const double *mg, void* f);

#endif // SOFA_GPU_CUDA_DOUBLE

}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void UniformMassCuda1t_addMDx_kernel(int size, const real mass, real* res, const real* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        res[index] += dx[index] * mass;
    }
}

template<class real>
__global__ void UniformMassCuda3t_addMDx_kernel(int size, const real mass, CudaVec3<real>* res, const CudaVec3<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //res[index] += dx[index] * mass;
        CudaVec3<real> dxi = dx[index];
        CudaVec3<real> ri = res[index];
        ri += dxi * mass;
        res[index] = ri;
    }
}

template<class real>
__global__ void UniformMassCuda3t1_addMDx_kernel(int size, const real mass, CudaVec4<real>* res, const CudaVec4<real>* dx)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //res[index] += dx[index] * mass;
        CudaVec4<real> dxi = dx[index];
        CudaVec4<real> ri = res[index];
        ri.x += dxi.x * mass;
        ri.y += dxi.y * mass;
        ri.z += dxi.z * mass;
        res[index] = ri;
    }
}

template<class real>
__global__ void UniformMassCudaRigid3t_addMDx_kernel(const unsigned int size, const real mass, real* res, const real* dx)
{
	__shared__ real tmp[6*BSIZE];
	
	int bid = 6*BSIZE*blockIdx.x;
	int gid = BSIZE*blockIdx.x + threadIdx.x;
	
	tmp[threadIdx.x] = dx[bid+threadIdx.x];
	tmp[BSIZE+threadIdx.x] = dx[BSIZE+bid+threadIdx.x];
	tmp[2*BSIZE+threadIdx.x] = dx[2*BSIZE+bid+threadIdx.x];
	tmp[3*BSIZE+threadIdx.x] = dx[3*BSIZE+bid+threadIdx.x];
	tmp[4*BSIZE+threadIdx.x] = dx[4*BSIZE+bid+threadIdx.x];
	tmp[5*BSIZE+threadIdx.x] = dx[5*BSIZE+bid+threadIdx.x];
	
	__syncthreads();
	
	if( gid < size )
	{
		tmp[6*threadIdx.x] *= mass;
		tmp[6*threadIdx.x+1] *= mass;
		tmp[6*threadIdx.x+2] *= mass;
		tmp[6*threadIdx.x+3] *= mass;
		tmp[6*threadIdx.x+4] *= mass;
		tmp[6*threadIdx.x+5] *= mass;
	}
	
	__syncthreads();
	
	res[bid+threadIdx.x] += tmp[threadIdx.x];
	res[BSIZE+bid+threadIdx.x] += tmp[BSIZE+threadIdx.x];
	res[2*BSIZE+bid+threadIdx.x] += tmp[2*BSIZE+threadIdx.x];
	res[3*BSIZE+bid+threadIdx.x] += tmp[3*BSIZE+threadIdx.x];
	res[4*BSIZE+bid+threadIdx.x] += tmp[4*BSIZE+threadIdx.x];
	res[5*BSIZE+bid+threadIdx.x] += tmp[5*BSIZE+threadIdx.x];
	
}

template<class real>
__global__ void UniformMassCuda1t_accFromF_kernel(int size, const real inv_mass, real* a, const real* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        a[index] = f[index] * inv_mass;
    }
}

template<class real>
__global__ void UniformMassCuda3t_accFromF_kernel(int size, const real inv_mass, CudaVec3<real>* a, const CudaVec3<real>* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //a[index] = f[index] * inv_mass;
        CudaVec3<real> fi = f[index];
        fi *= inv_mass;
        a[index] = fi;
    }
}

template<class real>
__global__ void UniformMassCuda3t1_accFromF_kernel(int size, const real inv_mass, CudaVec4<real>* a, const CudaVec4<real>* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //a[index] = f[index] * inv_mass;
        CudaVec4<real> fi = f[index];
        fi.x *= inv_mass;
        fi.y *= inv_mass;
        fi.z *= inv_mass;
        a[index] = fi;
    }
}


/**
 * 
 * */
template<class real>
__global__ void UniformMassCudaRigid3t_accFromF_kernel(const unsigned int size, const real inv_mass, real* a, const real* f)
{
	__shared__ real tmp[6*BSIZE];
	unsigned int bid = 6*BSIZE*blockIdx.x;
	

// 		real fi = f[gid] * inv_mass;
// 		a[gid] = fi;
	tmp[threadIdx.x] = f[bid+threadIdx.x];
	tmp[BSIZE+threadIdx.x] = f[BSIZE+bid+threadIdx.x];
	tmp[2*BSIZE+threadIdx.x] = f[2*BSIZE+bid+threadIdx.x];
	tmp[3*BSIZE+threadIdx.x] = f[3*BSIZE+bid+threadIdx.x];
	tmp[4*BSIZE+threadIdx.x] = f[4*BSIZE+bid+threadIdx.x];
	tmp[5*BSIZE+threadIdx.x] = f[5*BSIZE+bid+threadIdx.x];
	
	__syncthreads();
	
	if( (BSIZE*blockIdx.x + threadIdx.x) < size )
	{
		tmp[6*threadIdx.x] *= inv_mass;
		tmp[6*threadIdx.x+1] *= inv_mass;
		tmp[6*threadIdx.x+2] *= inv_mass;
		tmp[6*threadIdx.x+3] *= inv_mass;
		tmp[6*threadIdx.x+4] *= inv_mass;
		tmp[6*threadIdx.x+5] *= inv_mass;
	}
	
	__syncthreads();
	

	a[bid+threadIdx.x] = tmp[threadIdx.x];
	a[BSIZE+bid+threadIdx.x] = tmp[BSIZE+threadIdx.x];
	a[2*BSIZE+bid+threadIdx.x] = tmp[2*BSIZE+threadIdx.x];
	a[3*BSIZE+bid+threadIdx.x] = tmp[3*BSIZE+threadIdx.x];
	a[4*BSIZE+bid+threadIdx.x] = tmp[4*BSIZE+threadIdx.x];
	a[5*BSIZE+bid+threadIdx.x] = tmp[5*BSIZE+threadIdx.x];

}

template<class real>
__global__ void UniformMassCuda1t_addForce_kernel(int size, const real mg, real* f)
{
    int index = umul24(blockIdx.x,BSIZE);
    if (index < size)
    {
        f[index] += mg;
    }
}

template<class real>
//__global__ void UniformMassCuda3t_addForce_kernel(int size, const CudaVec3<real> mg, real* f)
__global__ void UniformMassCuda3t_addForce_kernel(int size, real mg_x, real mg_y, real mg_z, real* f)
{
    //int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    //f[index] += mg;
    f += umul24(blockIdx.x,BSIZE*3); //blockIdx.x*BSIZE*3;
    int index = threadIdx.x;
    __shared__  real temp[BSIZE*3];
    temp[index] = f[index];
    temp[index+BSIZE] = f[index+BSIZE];
    temp[index+2*BSIZE] = f[index+2*BSIZE];

    __syncthreads();

    if (umul24(blockIdx.x,BSIZE)+threadIdx.x < size)
    {
        int index3 = umul24(index,3); //3*index;
        temp[index3+0] += mg_x;
        temp[index3+1] += mg_y;
        temp[index3+2] += mg_z;
    }

    __syncthreads();

    f[index] = temp[index];
    f[index+BSIZE] = temp[index+BSIZE];
    f[index+2*BSIZE] = temp[index+2*BSIZE];
}

template<class real>
__global__ void UniformMassCuda3t1_addForce_kernel(int size, real mg_x, real mg_y, real mg_z, CudaVec4<real>* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
    {
        //f[index] += mg;
        CudaVec4<real> fi = f[index];
        fi.x += mg_x;
        fi.y += mg_y;
        fi.z += mg_z;
        f[index] = fi;
    }
}

template<class real>
__global__ void UniformMassCudaRigid3t_addForce_kernel(const unsigned int size, const real mg_x, const real mg_y, const real mg_z, real* f)
{
	__shared__ real tmp[6*BSIZE];
	const unsigned int bid = 6*BSIZE*blockIdx.x;
	
	tmp[threadIdx.x] = f[bid+threadIdx.x];
	tmp[BSIZE+threadIdx.x] = f[BSIZE+bid+threadIdx.x];
	tmp[2*BSIZE+threadIdx.x] = f[2*BSIZE+bid+threadIdx.x];
	tmp[3*BSIZE+threadIdx.x] = f[3*BSIZE+bid+threadIdx.x];
	tmp[4*BSIZE+threadIdx.x] = f[4*BSIZE+bid+threadIdx.x];
	tmp[5*BSIZE+threadIdx.x] = f[5*BSIZE+bid+threadIdx.x];
	
	__syncthreads();
	
	if( (BSIZE*blockIdx.x + threadIdx.x) < size )
	{
		tmp[6*threadIdx.x] += mg_x;
		tmp[6*threadIdx.x+1] += mg_y;
		tmp[6*threadIdx.x+2] += mg_z;
	}
	
	__syncthreads();
	

	f[bid+threadIdx.x] = tmp[threadIdx.x];
	f[BSIZE+bid+threadIdx.x] = tmp[BSIZE+threadIdx.x];
	f[2*BSIZE+bid+threadIdx.x] = tmp[2*BSIZE+threadIdx.x];
	f[3*BSIZE+bid+threadIdx.x] = tmp[3*BSIZE+threadIdx.x];
	f[4*BSIZE+bid+threadIdx.x] = tmp[4*BSIZE+threadIdx.x];
	f[5*BSIZE+bid+threadIdx.x] = tmp[5*BSIZE+threadIdx.x];	
// 	__shared__ real sf[6*BSIZE];
// 	unsigned int gid = BSIZE*blockIdx.x + threadIdx.x;
// 	
// 	sf[threadIdx.x] = f[gid];
// 	sf[BSIZE+threadIdx.x] = f[BSIZE+gid];
// 	sf[2*BSIZE+threadIdx.x] = f[2*BSIZE+gid];
// 	sf[3*BSIZE+threadIdx.x] = f[3*BSIZE+gid];
// 	sf[4*BSIZE+threadIdx.x] = f[4*BSIZE+gid];
// 	sf[5*BSIZE+threadIdx.x] = f[5*BSIZE+gid];
// 	
// 	__syncthreads();
// 	
// 	if( gid < size  )
// 	{
// 		sf[6*threadIdx.x] += mg_x;
// 		sf[6*threadIdx.x+1] += mg_y;
// 		sf[6*threadIdx.x+2] += mg_z;
// 	}
// 	
// 	__syncthreads();
// 	
// 
// 	f[gid] = sf[threadIdx.x];
// 	f[BSIZE+gid] = sf[BSIZE+threadIdx.x];
// 	f[2*BSIZE+gid] = sf[2*BSIZE+threadIdx.x];
// 	f[3*BSIZE+gid] = sf[3*BSIZE+threadIdx.x];
// 	f[4*BSIZE+gid] = sf[4*BSIZE+threadIdx.x];
// 	f[5*BSIZE+gid] = sf[5*BSIZE+threadIdx.x];
}

//////////////////////
// CPU-side methods //
//////////////////////

void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3t_addMDx_kernel<float><<< grid, threads >>>(size, mass, (CudaVec3<float>*)res, (const CudaVec3<float>*)dx);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda1t_addMDx_kernel<float><<< grid, threads >>>(3*size, mass, (float*)res, (const float*)dx); mycudaDebugError("UniformMassCuda1t_addMDx_kernel<float>");}
}

void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_addMDx_kernel<float><<< grid, threads >>>(size, mass, (CudaVec4<float>*)res, (const CudaVec4<float>*)dx); mycudaDebugError("UniformMassCuda3t1_addMDx_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1t_addMDx_kernel<float><<< grid, threads >>>(4*size, mass, (float*)res, (const float*)dx);
}

void UniformMassCudaRigid3f_addMDx(unsigned int size, float mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
	
	UniformMassCudaRigid3t_addMDx_kernel<float><<< grid, threads >>>(size, mass, (float*)res, (float*)dx);
	mycudaDebugError("UniformMassCudaRigid3t_addMDx_kernel<float>");
}

void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3t_accFromF_kernel<float><<< grid, threads >>>(size, 1.0f/mass, (CudaVec3<float>*)a, (const CudaVec3<float>*)f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda1t_accFromF_kernel<float><<< grid, threads >>>(3*size, 1.0f/mass, (float*)a, (const float*)f); mycudaDebugError("UniformMassCuda1t_accFromF_kernel<float>");}
}

void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_accFromF_kernel<float><<< grid, threads >>>(size, 1.0f/mass, (CudaVec4<float>*)a, (const CudaVec4<float>*)f); mycudaDebugError("UniformMassCuda3t1_accFromF_kernel<float>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1t_accFromF_kernel<float><<< grid, threads >>>(4*size, 1.0f/mass, (float*)a, (const float*)f);
}

void UniformMassCudaRigid3f_accFromF(unsigned int size, float mass, void* a, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
	
	UniformMassCudaRigid3t_accFromF_kernel<float><<< grid, threads >>>(size, 1.0f/mass, (float*)a, (float*)dx);
	
	mycudaDebugError("UniformMassCudaRigid3f_accFromF");
}

void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t_addForce_kernel<float><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (float*)f); mycudaDebugError("UniformMassCuda3t_addForce_kernel<float>");}
}

void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_addForce_kernel<float><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (CudaVec4<float>*)f); mycudaDebugError("UniformMassCuda3t1_addForce_kernel<float>");}
}

void UniformMassCudaRigid3f_addForce(unsigned int size, const float* mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
// 	sofa::gpu::cuda::mycudaPrintf("mg = %f %f %f\n", mg[0], mg[1], mg[2]);
	UniformMassCudaRigid3t_addForce_kernel<float><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (float*)f);
	
	mycudaDebugError("UniformMassCudaRigid3f_addForce");
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void UniformMassCuda3d_addMDx(unsigned int size, double mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3t_addMDx_kernel<double><<< grid, threads >>>(size, mass, (CudaVec3<double>*)res, (const CudaVec3<double>*)dx);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda1t_addMDx_kernel<double><<< grid, threads >>>(3*size, mass, (double*)res, (const double*)dx); mycudaDebugError("UniformMassCuda1t_addMDx_kernel<double>");}
}

void UniformMassCuda3d1_addMDx(unsigned int size, double mass, void* res, const void* dx)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_addMDx_kernel<double><<< grid, threads >>>(size, mass, (CudaVec4<double>*)res, (const CudaVec4<double>*)dx); mycudaDebugError("UniformMassCuda3t1_addMDx_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1t_addMDx_kernel<double><<< grid, threads >>>(4*size, mass, (double*)res, (const double*)dx);
}

void UniformMassCuda3d_accFromF(unsigned int size, double mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    //dim3 grid((size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda3t_accFromF_kernel<double><<< grid, threads >>>(size, 1.0f/mass, (CudaVec3<double>*)a, (const CudaVec3<double>*)f);
    dim3 grid((3*size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda1t_accFromF_kernel<double><<< grid, threads >>>(3*size, 1.0f/mass, (double*)a, (const double*)f); mycudaDebugError("UniformMassCuda1t_accFromF_kernel<double>");}
}

void UniformMassCuda3d1_accFromF(unsigned int size, double mass, void* a, const void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_accFromF_kernel<double><<< grid, threads >>>(size, 1.0f/mass, (CudaVec4<double>*)a, (const CudaVec4<double>*)f); mycudaDebugError("UniformMassCuda3t1_accFromF_kernel<double>");}
    //dim3 grid((4*size+BSIZE-1)/BSIZE,1);
    //UniformMassCuda1t_accFromF_kernel<double><<< grid, threads >>>(4*size, 1.0f/mass, (double*)a, (const double*)f);
}

void UniformMassCuda3d_addForce(unsigned int size, const double *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t_addForce_kernel<double><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (double*)f); mycudaDebugError("UniformMassCuda3t_addForce_kernel<double>");}
}

void UniformMassCuda3d1_addForce(unsigned int size, const double *mg, void* f)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {UniformMassCuda3t1_addForce_kernel<double><<< grid, threads >>>(size, mg[0], mg[1], mg[2], (CudaVec4<double>*)f); mycudaDebugError("UniformMassCuda3t1_addForce_kernel<double>");}
}

#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
