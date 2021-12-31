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
#include <SofaMiscFem/FastTetrahedralCorotationalForceField.h>

namespace sofa::component::forcefield
{

template <>
class FastTetrahedralCorotationalForceFieldData<gpu::cuda::CudaVec3Types>
{
public:
    typedef FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3Types> Main;
    
    struct GPUTetrahedron
    {
        int indices[4];
    };

    typedef gpu::cuda::CudaVector<GPUTetrahedron> VecGPUTetrahedron;

    VecGPUTetrahedron gpuTetrahedra;

    void reinit(Main* m)
    {

        const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = m->l_topology.get()->getTetrahedra();
        helper::WriteAccessor< VecGPUTetrahedron > _gpuTetrahedra = this->gpuTetrahedra;

        _gpuTetrahedra.resize(tetrahedra.size());
        for (unsigned int i=0;i< tetrahedra.size();++i)
        {
            for (unsigned int j = 0; j < 4; ++j)
            {
                _gpuTetrahedra[i].indices[j] = tetrahedra[i][j];
            }
        }
    }

};

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);


#ifdef SOFA_GPU_CUDA_DOUBLE
template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::forcefield
