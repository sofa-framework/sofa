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

#include <sofa/gpu/cuda/CudaFastTetrahedralCorotationalForceField.h>
#include <SofaMiscFem/FastTetrahedralCorotationalForceField.inl>

namespace sofa::gpu::cuda
{


extern "C"
{
void FastTetrahedralCorotationalForceFieldCuda3f_addForce(unsigned int size, void* f, const void* x, const void* v,
    unsigned int nbTetrahedra, void* tetrahedronInfo, const void* gpuTetra);

void FastTetrahedralCorotationalForceFieldCuda3f_computeEdgeMatrices(unsigned int nbTetrahedra, const void* tetrahedronInf, void* edgeDfDx, const void* gpuTetra);

void FastTetrahedralCorotationalForceFieldCuda3f_addDForce(unsigned int nbedges, void* df, const void* dx, float kFactor,
    const void* edgeDfDx, const void* gpuEdges);

#ifdef SOFA_GPU_CUDA_DOUBLE
void FastTetrahedralCorotationalForceFieldCuda3d_addForce(unsigned int size, void* f, const void* x, const void* v,
    unsigned int nbTetrahedra, void* tetrahedronInfo, const void* gpuTetra);

void FastTetrahedralCorotationalForceFieldCuda3d_computeEdgeMatrices(unsigned int nbTetrahedra, const void* tetrahedronInf, void* edgeDfDx, const void* gpuTetra);

void FastTetrahedralCorotationalForceFieldCuda3d_addDForce(unsigned int size, void* f, const void* dx, double kFactor,
    const void* triangleState, const void* triangleInfo,
    unsigned int nbTriangles,
    const void* gpuTriangleInfo,
    double gamma, double mu); //, const void* dfdx);
#endif // SOFA_GPU_CUDA_DOUBLE

}

} // namespace sofa::gpu::cuda

namespace sofa::component::forcefield 
{

using namespace gpu::cuda;

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    sofa::Size nbTetrahedra = m_topology->getNbTetrahedra();
    VecTetrahedronRestInformation& tetrahedronInf = *tetrahedronInfo.beginEdit();
    const ExtraData::VecGPUTetrahedron& gpuTetra = m_data.gpuTetrahedra;

    f.resize(x.size());
 
    FastTetrahedralCorotationalForceFieldCuda3f_addForce(x.size(), f.deviceWrite(), x.deviceRead(), v.deviceRead(),
        nbTetrahedra, 
        tetrahedronInf.deviceWrite(),
        gpuTetra.deviceRead());


    updateMatrix = true; // next time assemble the matrix

    tetrahedronInfo.endEdit();
    d_f.endEdit();
}

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    const auto kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    
    const ExtraData::VecGPUTetrahedron& gpuTetra = m_data.gpuTetrahedra;

    if (updateMatrix == true)
    {
        sofa::Size nbTetrahedra = m_topology->getNbTetrahedra();
        const VecTetrahedronRestInformation& tetrahedronInf = tetrahedronInfo.getValue();
        VecMat3x3& edgeDfDx = *edgeInfo.beginEdit();

        // reset all edge matrices
        for (unsigned int j = 0; j < edgeDfDx.size(); j++)
        {
            edgeDfDx[j].clear();
        }

        FastTetrahedralCorotationalForceFieldCuda3f_computeEdgeMatrices(nbTetrahedra, tetrahedronInf.deviceRead(), edgeDfDx.deviceWrite(), gpuTetra.deviceRead());

        updateMatrix = false;
        edgeInfo.endEdit();
    }


    const VecMat3x3& edgeDfDx = edgeInfo.getValue();
    const ExtraData::VecGPUEdge& gpuEdges = m_data.gpuEdges;
    sofa::Size nbedges = m_topology->getNbEdges();
    FastTetrahedralCorotationalForceFieldCuda3f_addDForce(nbedges, df.deviceWrite(), dx.deviceRead(), kFactor,
        edgeDfDx.deviceRead(),
        gpuEdges.deviceRead());
    
    d_df.endEdit();
}


#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    sofa::Size nbTetrahedra = m_topology->getNbTetrahedra();
    VecTetrahedronRestInformation& tetrahedronInf = *tetrahedronInfo.beginEdit();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = m_topology->getTetrahedra();
    const ExtraData::VecGPUTetrahedron& gpuTetra = m_data.gpuTetrahedra;

    f.resize(x.size());

    FastTetrahedralCorotationalForceFieldCuda3d_addForce(x.size(), f.deviceWrite(), x.deviceRead(), v.deviceRead(),
        nbTetrahedra,
        tetrahedronInf.deviceWrite(),
        gpuTetra.deviceRead());


    updateMatrix = true; // next time assemble the matrix

    tetrahedronInfo.endEdit();
    d_f.endEdit();
}

template <>
void FastTetrahedralCorotationalForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const VecTriangleState& triState = d_triangleState.getValue();
    const VecTriangleInfo& triInfo = d_triangleInfo.getValue();
    const unsigned int nbTriangles = m_topology->getNbTriangles();
    const InternalData::VecGPUTriangleInfo& gpuTriangleInfo = data.gpuTriangleInfo;
    const Real gamma = this->gamma;
    const Real mu = this->mu;

    df.resize(dx.size());

    FastTetrahedralCorotationalForceFieldCuda3d_addDForce(dx.size(), df.deviceWrite(), dx.deviceRead(), kFactor,
        triState.deviceRead(),
        triInfo.deviceRead(),
        nbTriangles,
        gpuTriangleInfo.deviceRead(),
        gamma, mu);

    d_df.endEdit();
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::forcefield
