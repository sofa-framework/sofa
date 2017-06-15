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
#ifndef SOFA_GPU_CUDA_CUDATRIANGULARFEMFORCEFIELDOPTIM_H
#define SOFA_GPU_CUDA_CUDATRIANGULARFEMFORCEFIELDOPTIM_H

#include "CudaTypes.h"
#include <SofaGeneralSimpleFem/TriangularFEMForceFieldOptim.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <>
class TriangularFEMForceFieldOptimInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    typedef TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3fTypes> Main;
    struct GPUTriangleInfo
    {
        int ia, ib, ic;
    };

    typedef gpu::cuda::CudaVector<GPUTriangleInfo> VecGPUTriangleInfo;

    VecGPUTriangleInfo gpuTriangleInfo;

    void reinit(Main* m)
    {
        const Main::VecElement& triangles = m->_topology->getTriangles();
        helper::WriteAccessor< VecGPUTriangleInfo > gpuTriangleInfo = this->gpuTriangleInfo;

        gpuTriangleInfo.resize(triangles.size());
        for (unsigned int i=0;i<triangles.size();++i)
        {
            gpuTriangleInfo[i].ia = triangles[i][0];
            gpuTriangleInfo[i].ib = triangles[i][1];
            gpuTriangleInfo[i].ic = triangles[i][2];
        }
        std::cout << "CREATED " << gpuTriangleInfo.size() << " GPU TRIANGLESTATE" << std::endl;
    }

};

template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
