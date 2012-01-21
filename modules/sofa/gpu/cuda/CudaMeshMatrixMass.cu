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
    void MeshMatrixMassCuda_addMDxf(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res);

//        void MeshMatrixMassCuda_addMToMatrixf(unsigned int size, float factor, float massLumpingCoeff, const void * mass, const void* dx, void* res);
}


template<class real>
__global__ void MeshMatrixMassCuda_addMDx_kernel(int size, real factor, real massLumpingCoeff,const real * vertexMass, const real* dx, real* res)
{
    int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
    int index2 = index * 2;

    //LUMPING INTEGRATION METHOD-------------------------------
    if (index < size)
    {
        res[index2+0] += dx[index2+0] * vertexMass[index] * massLumpingCoeff * factor;
        res[index2+1] += dx[index2+1] * vertexMass[index] * massLumpingCoeff * factor;
    }
}

//template<class real>
//__global__ void MeshMatrixMassCuda_addMToMatrix_kernel(int size,real factor, real massLumpingCoeff, real factor,const real * mass, const real* dx, real* res) {
//        int index = umul24(blockIdx.x,BSIZE)+threadIdx.x;
//        int index3 = index * 3;
//        if (index < size) {

//          calc(r.matrix, vertexMass[i] * massLumpingCoeff, r.offset + N*i, mFactor);

//          for (int i=0;i<N;++i)
//              mat->add(pos+i, pos+i, m);
//          mat->add(pos+N, pos+N, mass.inertiaMassMatrix*fact);

//          res[index3+0] += dx[index3+0] * mass[index] * factor;
//          res[index3+1] += dx[index3+1] * mass[index] * factor;
//          res[index3+2] += dx[index3+2] * mass[index] * factor;
//        }
//}

//////////////////////
// CPU-side methods //
//////////////////////

void MeshMatrixMassCuda_addMDxf(unsigned int size, float factor, float massLumpingCoeff, const void * vertexMass, const void* dx, void* res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    {MeshMatrixMassCuda_addMDx_kernel<float><<< grid, threads >>>(size, factor, massLumpingCoeff, (const float *) vertexMass, (const float *) dx, (float*) res);}
}

//void MeshMatrixMassCuda_addMToMatrixf(unsigned int size, float factor, float massLumpingCoeff, const void * mass, const void* dx, void* res) {
//        dim3 threads(BSIZE,1);
//        dim3 grid((size+BSIZE-1)/BSIZE,1);
//        {MeshMatrixMassCuda_addMToMatrix_kernel<float><<< grid, threads >>>(size,factor, massLumpingCoeff, (const float *) mass, (const float*)dx, (float*)res); mycudaDebugError("MeshMatrixMassCuda_addMDx_kernel<float>");}
//}

#if defined(__cplusplus) && CUDA_VERSION < 2000
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
