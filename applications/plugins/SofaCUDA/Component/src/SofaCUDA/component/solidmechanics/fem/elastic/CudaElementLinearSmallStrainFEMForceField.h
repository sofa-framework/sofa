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
#include <sofa/component/solidmechanics/fem/elastic/ElementLinearSmallStrainFEMForceField.h>

namespace sofa::gpu::cuda
{

extern "C"
{
    void ElementLinearSmallStrainFEMForceFieldCuda3f_addForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* stiffness,
        const void* x,
        const void* x0,
        void* f,
        void* eforce,
        const void* velems);

    void ElementLinearSmallStrainFEMForceFieldCuda3f_addDForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* stiffness,
        const void* dx,
        void* df,
        void* eforce,
        const void* velems,
        float kFactor);

    void ElementLinearSmallStrainFEMForceFieldCuda3d_addForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* stiffness,
        const void* x,
        const void* x0,
        void* f,
        void* eforce,
        const void* velems);

    void ElementLinearSmallStrainFEMForceFieldCuda3d_addDForce(
        unsigned int nbElem,
        unsigned int nbVertex,
        unsigned int nbNodesPerElem,
        unsigned int maxElemPerVertex,
        const void* elements,
        const void* stiffness,
        const void* dx,
        void* df,
        void* eforce,
        const void* velems,
        double kFactor);
}

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * CUDA-accelerated version of ElementLinearSmallStrainFEMForceField.
 *
 * Works with any element type (Edge, Triangle, Quad, Tetrahedron, Hexahedron).
 * Both addForce and addDForce run entirely on GPU.
 *
 * Uses a two-kernel approach for addDForce:
 *   Kernel 1: compute per-element forces (1 thread/element, fully unrolled)
 *   Kernel 2: gather per-vertex (1 thread/vertex, no atomics)
 *
 * Compared to the corotational version, no rotation matrices are needed.
 */
template<class DataTypes, class ElementType>
class CudaElementLinearSmallStrainFEMForceField
    : public ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE2(CudaElementLinearSmallStrainFEMForceField, DataTypes, ElementType),
        SOFA_TEMPLATE2(ElementLinearSmallStrainFEMForceField, DataTypes, ElementType));

    using Real = sofa::Real_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;

    static const std::string GetCustomClassName()
    {
        return ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::GetCustomClassName();
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

    CudaElementLinearSmallStrainFEMForceField() = default;

    void uploadStiffnessAndConnectivity();

    gpu::cuda::CudaVector<Real> m_gpuStiffness;      ///< Symmetric block-format stiffness per element
    gpu::cuda::CudaVector<int>  m_gpuElements;        ///< SoA connectivity: elements[nodeIdx * nbElem + elemId]
    gpu::cuda::CudaVector<Real> m_gpuElementForce;    ///< Intermediate per-element per-node force buffer
    gpu::cuda::CudaVector<int>  m_gpuVelems;          ///< SoA vertex-to-element mapping, 0-terminated

    unsigned int m_maxElemPerVertex = 0;
    unsigned int m_nbVertices = 0;

    bool m_gpuDataUploaded = false;
};

} // namespace sofa::component::solidmechanics::fem::elastic
