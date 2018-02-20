/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
    {ExternalForceFieldCuda3f_addForce_kernel<<< grid, threads, BSIZE*3*sizeof(float) >>>(size, (float*)f, (const unsigned*)indices,(const float*)forces); mycudaDebugError("ExternalForceFieldCuda3f_addForce_kernel");}
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
