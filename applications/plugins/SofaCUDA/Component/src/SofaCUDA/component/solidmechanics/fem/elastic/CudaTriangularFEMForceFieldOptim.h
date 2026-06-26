/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceFieldOptim.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class TCoord, class TDeriv, class TReal>
class TriangularFEMForceFieldOptimInternalData< gpu::cuda::CudaVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord, TDeriv, TReal> DataTypes;
    typedef TriangularFEMForceFieldOptim<DataTypes> Main;
    
    struct GPUTriangleInfo
    {
        int ia, ib, ic;
    };

    typedef gpu::cuda::CudaVector<GPUTriangleInfo> VecGPUTriangleInfo;

    VecGPUTriangleInfo gpuTriangleInfo;

    void reinit(Main* m)
    {

        const typename Main::VecElement& triangles = m->l_topology.get()->getTriangles();
        helper::WriteAccessor< VecGPUTriangleInfo > gpuTriangleInfo = this->gpuTriangleInfo;

        gpuTriangleInfo.resize(triangles.size());
        for (unsigned int i=0;i<triangles.size();++i)
        {
            gpuTriangleInfo[i].ia = triangles[i][0];
            gpuTriangleInfo[i].ib = triangles[i][1];
            gpuTriangleInfo[i].ic = triangles[i][2];
        }
    }

};

template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);


#ifdef SOFA_GPU_CUDA_DOUBLE
template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
template <>
void TriangularFEMForceFieldOptim<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::solidmechanics::fem::elastic
