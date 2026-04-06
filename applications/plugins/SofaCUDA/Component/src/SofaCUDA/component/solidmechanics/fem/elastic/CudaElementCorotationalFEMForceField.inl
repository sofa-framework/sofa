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
#include <SofaCUDA/component/solidmechanics/fem/elastic/CudaElementCorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/ElementCorotationalFEMForceField.inl>
#include <sofa/core/behavior/ForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::init()
{
    ElementCorotationalFEMForceField<DataTypes, ElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        uploadStiffnessAndConnectivity();
    }
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::uploadStiffnessAndConnectivity()
{
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;

    if (!this->l_topology) return;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto& assembledMatrices = this->m_assembledStiffnessMatrices;

    const auto nbElem = elements.size();
    constexpr auto nDofs = trait::NumberOfDofsInElement;
    constexpr auto nNodes = trait::NumberOfNodesInElement;

    // Upload stiffness matrices (flat row-major NxN per element)
    m_gpuStiffness.resize(nbElem * nDofs * nDofs);
    {
        auto* dst = m_gpuStiffness.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& K = assembledMatrices[e];
            for (unsigned int i = 0; i < nDofs; ++i)
                for (unsigned int j = 0; j < nDofs; ++j)
                    dst[e * nDofs * nDofs + i * nDofs + j] = static_cast<float>(K[i][j]);
        }
    }

    // Upload element connectivity (nNodes node indices per element)
    m_gpuElements.resize(nbElem * nNodes);
    {
        auto* dst = m_gpuElements.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& element = elements[e];
            for (unsigned int n = 0; n < nNodes; ++n)
                dst[e * nNodes + n] = static_cast<int>(element[n]);
        }
    }

    m_gpuDataUploaded = true;
    m_gpuRotationsUploaded = false;
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::uploadRotations()
{
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    constexpr auto dim = trait::spatial_dimensions;

    const auto& rotations = this->m_rotations;
    const auto nbElem = rotations.size();

    m_gpuRotations.resize(nbElem * dim * dim);
    {
        auto* dst = m_gpuRotations.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& R = rotations[e];
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    dst[e * dim * dim + i * dim + j] = static_cast<float>(R[i][j]);
        }
    }

    m_gpuRotationsUploaded = true;
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::addForce(
    const sofa::core::MechanicalParams* mparams,
    sofa::DataVecDeriv_t<DataTypes>& f,
    const sofa::DataVecCoord_t<DataTypes>& x,
    const sofa::DataVecDeriv_t<DataTypes>& v)
{
    // Run on CPU: computes rotations and forces
    ElementCorotationalFEMForceField<DataTypes, ElementType>::addForce(mparams, f, x, v);

    // Upload the freshly-computed rotations to GPU for subsequent addDForce calls
    uploadRotations();
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::addDForce(
    const sofa::core::MechanicalParams* mparams,
    sofa::DataVecDeriv_t<DataTypes>& d_df,
    const sofa::DataVecDeriv_t<DataTypes>& d_dx)
{
    if (this->isComponentStateInvalid())
        return;

    if (!m_gpuDataUploaded || !m_gpuRotationsUploaded)
    {
        // Fallback to CPU if GPU data not ready
        ElementCorotationalFEMForceField<DataTypes, ElementType>::addDForce(mparams, d_df, d_dx);
        return;
    }

    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    if (df.size() < dx.size())
        df.resize(dx.size());

    const auto kFactor = static_cast<float>(
        sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
            mparams, this->rayleighStiffness.getValue()));

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto nbElem = static_cast<unsigned int>(elements.size());

    gpu::cuda::ElementCorotationalFEMForceFieldCuda3f_addDForce(
        nbElem,
        trait::NumberOfNodesInElement,
        trait::NumberOfDofsInElement,
        trait::spatial_dimensions,
        m_gpuElements.deviceRead(),
        m_gpuRotations.deviceRead(),
        m_gpuStiffness.deviceRead(),
        dx.deviceRead(),
        df.deviceWrite(),
        kFactor);

    d_df.endEdit();
}

} // namespace sofa::component::solidmechanics::fem::elastic
