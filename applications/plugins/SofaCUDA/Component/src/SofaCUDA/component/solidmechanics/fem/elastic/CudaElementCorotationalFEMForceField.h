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
    void ElementCorotationalFEMForceFieldCuda3f_addForceWithRotations(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* initRotTransposed,
        const void* stiffness,
        const void* x,
        const void* x0,
        void* f,
        void* eforce,
        void* rotationsOut,
        const void* velems);

    void ElementCorotationalFEMForceFieldCuda3f_addForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* rotations,
        const void* stiffness,
        const void* x,
        const void* x0,
        void* f,
        void* eforce,
        const void* velems);

    void ElementCorotationalFEMForceFieldCuda3f_addDForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* rotations,
        const void* stiffness,
        const void* dx,
        void* df,
        void* eforce,
        const void* velems,
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
 *
 * Uses a two-kernel approach for addDForce:
 *   Kernel 1: compute per-element forces (1 thread/element, fully unrolled)
 *   Kernel 2: gather per-vertex (1 thread/vertex, no atomics)
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

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

protected:

    CudaElementCorotationalFEMForceField() = default;

    void uploadStiffnessAndConnectivity();
    void uploadRotations();
    void uploadInitialRotationsTransposed();
    void downloadRotations();

    gpu::cuda::CudaVector<float> m_gpuStiffness;                  ///< Symmetric block-format stiffness per element
    gpu::cuda::CudaVector<float> m_gpuRotations;                  ///< Flat 3x3 rotation matrices per element
    gpu::cuda::CudaVector<float> m_gpuInitialRotationsTransposed; ///< Flat 3x3 initial rotation transposed per element
    gpu::cuda::CudaVector<int>   m_gpuElements;                   ///< SoA connectivity: elements[nodeIdx * nbElem + elemId]
    gpu::cuda::CudaVector<float> m_gpuElementForce;               ///< Intermediate per-element per-node force buffer
    gpu::cuda::CudaVector<int>   m_gpuVelems;                     ///< SoA vertex-to-element mapping, 0-terminated

    unsigned int m_maxElemPerVertex = 0;
    unsigned int m_nbVertices = 0;

    bool m_gpuDataUploaded = false;
    bool m_gpuRotationsUploaded = false;
    bool m_gpuRotationMethodSupported = false;
};

} // namespace sofa::component::solidmechanics::fem::elastic
