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
#include "CudaMathRigid.h"

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
    void LinearForceFieldCudaRigid3f_addForce(unsigned size, const void* indices, const void* forces, void* f);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void LinearForceFieldCudaRigid3d_addForce(unsigned size, const void* indices, const void *forces, void* f);
#endif
}// extern "C"


//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void LinearForceFieldCudaRigid3t_addForce_kernel(unsigned size, const int* indices, real fx, real fy, real fz, real frx, real fry, real frz, CudaRigidDeriv3<real>* f)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;

    CudaRigidDeriv3<real> force = CudaRigidDeriv3<real>::make(fx, fy, fz, frx, fry, frz);
    if (index < size)
    {
        f[indices[index]] += force;
    }
}// addForce_kernel

//////////////////////
// CPU-side methods //
//////////////////////

void LinearForceFieldCudaRigid3f_addForce(unsigned size, const void* indices, const void* forces, void* f)
{
//	float* fv = (float*)forces;
// 	mycudaPrintf("Add : %f %f %f %f %f %f\n", fv[0], fv[1], fv[2], fv[3], fv[4], fv[5]);
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    LinearForceFieldCudaRigid3t_addForce_kernel<float>
    <<< grid, threads >>>
    (size, (const int*)indices, ((float*)forces)[0], ((float*)forces)[1], ((float*)forces)[2], ((float*)forces)[3], ((float*)forces)[4], ((float*)forces)[5], (CudaRigidDeriv3<float>*)f);
}// LinearForceFieldCudaRigid3f_addForce

#ifdef SOFA_GPU_CUDA_DOUBLE
void LinearForceFieldCudaRigid3d_addForce(unsigned size, const void* indices, const void* forces, void* f)
{
    dim3 threads(BSIZE, 1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    LinearForceFieldCudaRigid3t_addForce_kernel<double>
    <<< grid, threads >>>
    (size, (const int*)indices, ((double*)f)[0], ((double*)f)[1], ((double*)f)[2], ((double*)f)[3], ((double*)f)[4], ((double*)f)[5], (CudaRigidDeriv3<double>*)f);
}// LinearForceFieldCudaRigid3f_addForce
#endif // SOFA_GPU_CUDA_DOUBLE

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
