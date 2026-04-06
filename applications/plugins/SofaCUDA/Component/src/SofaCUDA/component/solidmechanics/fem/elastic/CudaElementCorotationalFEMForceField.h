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
#include <sofa/component/solidmechanics/fem/elastic/ElementCorotationalFEMForceField.h>

namespace sofa::gpu::cuda
{

extern "C"
{
    void ElementCorotationalFEMForceFieldCuda3f_addDForce(
        unsigned int nbElem,
        unsigned int nbNodesPerElem,
        unsigned int nbDofsPerElem,
        unsigned int spatialDim,
        const void* elements,
        const void* rotations,
        const void* stiffness,
        const void* dx,
        void* df,
        float kFactor);
}

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * CUDA-accelerated version of ElementCorotationalFEMForceField.
 *
 * Works with any element type (Edge, Triangle, Quad, Tetrahedron, Hexahedron).
 * The addDForce method (the CG hot path, called ~250 times per timestep) runs entirely on GPU.
 * The addForce method delegates to the CPU parent and uploads rotations to GPU afterwards.
 */
template<class DataTypes, class ElementType>
class CudaElementCorotationalFEMForceField
    : public ElementCorotationalFEMForceField<DataTypes, ElementType>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE2(CudaElementCorotationalFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(ElementCorotationalFEMForceField, DataTypes, ElementType));

    using Real = sofa::Real_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;

    static const std::string GetCustomClassName()
    {
        return ElementCorotationalFEMForceField<DataTypes, ElementType>::GetCustomClassName();
    }

    static const std::string GetCustomTemplateName()
    {
        return DataTypes::Name();
    }

    void init() override;

    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

    void addDForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& df,
        const sofa::DataVecDeriv_t<DataTypes>& dx) override;

protected:

    CudaElementCorotationalFEMForceField() = default;

    void uploadStiffnessAndConnectivity();
    void uploadRotations();

    gpu::cuda::CudaVector<float> m_gpuStiffness;   ///< Flat NxN stiffness matrices per element (N = nbDofsPerElement)
    gpu::cuda::CudaVector<float> m_gpuRotations;    ///< Flat DxD rotation matrices per element (D = spatial_dimensions)
    gpu::cuda::CudaVector<int>   m_gpuElements;     ///< Node indices per element

    bool m_gpuDataUploaded = false;
    bool m_gpuRotationsUploaded = false;
};

} // namespace sofa::component::solidmechanics::fem::elastic
