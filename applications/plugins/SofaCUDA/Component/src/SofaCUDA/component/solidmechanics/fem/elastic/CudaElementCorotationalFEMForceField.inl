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
#include <cstring>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::init()
{
    ElementCorotationalFEMForceField<DataTypes, ElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        uploadStiffnessAndConnectivity();
        uploadInitialRotationsTransposed();
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
    constexpr auto nNodes = trait::NumberOfNodesInElement;
    constexpr auto dim = trait::spatial_dimensions;

    // Find number of vertices
    unsigned int maxNodeId = 0;
    for (std::size_t e = 0; e < nbElem; ++e)
    {
        const auto& element = elements[e];
        for (unsigned int n = 0; n < nNodes; ++n)
        {
            if (static_cast<unsigned int>(element[n]) > maxNodeId)
                maxNodeId = static_cast<unsigned int>(element[n]);
        }
    }
    m_nbVertices = maxNodeId + 1;

    // Upload stiffness matrices in symmetric upper-triangle block format:
    // Only blocks (ni, nj) with nj >= ni are stored.
    // symIdx = ni * nNodes - ni*(ni-1)/2 + (nj - ni)
    // K[symIdx * dim * dim + di * dim + dj] per element
    constexpr auto nSymBlocks = nNodes * (nNodes + 1) / 2;
    m_gpuStiffness.resize(nbElem * nSymBlocks * dim * dim);
    {
        auto* dst = m_gpuStiffness.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& K = assembledMatrices[e];
            for (unsigned int ni = 0; ni < nNodes; ++ni)
            {
                const unsigned int diagIdx = ni * nNodes - ni * (ni - 1) / 2;
                for (unsigned int nj = ni; nj < nNodes; ++nj)
                {
                    const unsigned int symIdx = diagIdx + (nj - ni);
                    for (unsigned int di = 0; di < dim; ++di)
                        for (unsigned int dj = 0; dj < dim; ++dj)
                            dst[e * nSymBlocks * dim * dim
                                + symIdx * dim * dim
                                + di * dim + dj]
                                = static_cast<Real>(K[ni * dim + di][nj * dim + dj]);
                }
            }
        }
    }

    // Upload element connectivity in SoA layout:
    // elements[nodeIdx * nbElem + elemId] = global node index
    // Adjacent threads access adjacent memory for coalesced reads.
    m_gpuElements.resize(nNodes * nbElem);
    {
        auto* dst = m_gpuElements.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& element = elements[e];
            for (unsigned int n = 0; n < nNodes; ++n)
                dst[n * nbElem + e] = static_cast<int>(element[n]);
        }
    }

    // Build vertex-to-element mapping (velems)
    // For each vertex, stores the list of (elemId * nNodes + localNode + 1).
    // 0 is used as sentinel. SoA layout: velems[slot * nbVertex + vertexId].
    std::vector<std::vector<int>> vertexElems(m_nbVertices);
    for (std::size_t e = 0; e < nbElem; ++e)
    {
        const auto& element = elements[e];
        for (unsigned int n = 0; n < nNodes; ++n)
        {
            const int nodeId = static_cast<int>(element[n]);
            vertexElems[nodeId].push_back(
                static_cast<int>(e * nNodes + n + 1));
        }
    }

    m_maxElemPerVertex = 0;
    for (const auto& ve : vertexElems)
    {
        if (ve.size() > m_maxElemPerVertex)
            m_maxElemPerVertex = static_cast<unsigned int>(ve.size());
    }

    m_gpuVelems.resize(m_maxElemPerVertex * m_nbVertices);
    {
        auto* dst = m_gpuVelems.hostWrite();
        std::memset(dst, 0, m_maxElemPerVertex * m_nbVertices * sizeof(int));
        for (std::size_t v = 0; v < m_nbVertices; ++v)
        {
            for (std::size_t s = 0; s < vertexElems[v].size(); ++s)
                dst[s * m_nbVertices + v] = vertexElems[v][s];
        }
    }

    // Allocate intermediate per-element force buffer
    m_gpuElementForce.resize(nbElem * nNodes * dim);

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
                    dst[e * dim * dim + i * dim + j] = static_cast<Real>(R[i][j]);
        }
    }

    m_gpuRotationsUploaded = true;
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::uploadInitialRotationsTransposed()
{
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    constexpr auto dim = trait::spatial_dimensions;
    constexpr auto nNodes = trait::NumberOfNodesInElement;

    const auto& initRotT = this->m_initialRotationsTransposed;
    const auto nbElem = initRotT.size();
    if (nbElem == 0) return;

    m_gpuInitialRotationsTransposed.resize(nbElem * dim * dim);
    m_gpuRotations.resize(nbElem * dim * dim);
    {
        auto* dst = m_gpuInitialRotationsTransposed.hostWrite();
        for (std::size_t e = 0; e < nbElem; ++e)
        {
            const auto& R = initRotT[e];
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    dst[e * dim * dim + i * dim + j] = static_cast<Real>(R[i][j]);
        }
    }

    // Check if the rotation method is GPU-compatible
    const auto rotationMethodKey = this->m_rotationMethods.d_rotationMethod.getValue().key();
    m_gpuRotationMethodSupported = (nNodes >= 3)
        && (rotationMethodKey == "triangle" || rotationMethodKey == "hexahedron");
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::downloadRotations()
{
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    constexpr auto dim = trait::spatial_dimensions;

    if (!m_gpuRotationsUploaded) return;

    const auto nbElem = m_gpuRotations.size() / (dim * dim);
    this->m_rotations.resize(nbElem);

    const auto* src = m_gpuRotations.hostRead();
    for (std::size_t e = 0; e < nbElem; ++e)
    {
        auto& R = this->m_rotations[e];
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                R[i][j] = static_cast<Real>(src[e * dim * dim + i * dim + j]);
    }
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (m_gpuRotationMethodSupported && m_gpuRotationsUploaded)
        downloadRotations();

    ElementCorotationalFEMForceField<DataTypes, ElementType>::buildStiffnessMatrix(matrix);
}

template<class DataTypes, class ElementType>
void CudaElementCorotationalFEMForceField<DataTypes, ElementType>::addForce(
    const sofa::core::MechanicalParams* mparams,
    sofa::DataVecDeriv_t<DataTypes>& d_f,
    const sofa::DataVecCoord_t<DataTypes>& d_x,
    const sofa::DataVecDeriv_t<DataTypes>& d_v)
{
    if (this->isComponentStateInvalid())
        return;

    if (!m_gpuDataUploaded)
    {
        ElementCorotationalFEMForceField<DataTypes, ElementType>::addForce(mparams, d_f, d_x, d_v);
        uploadRotations();
        return;
    }

    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    constexpr auto nNodes = trait::NumberOfNodesInElement;
    constexpr auto dim = trait::spatial_dimensions;

    const VecCoord& x = d_x.getValue();
    auto restPositionAccessor = this->mstate->readRestPositions();
    const VecCoord& x0 = restPositionAccessor.ref();

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto nbElem = static_cast<unsigned int>(elements.size());
    const auto nbVertex = static_cast<unsigned int>(x.size());

    VecDeriv& f = *d_f.beginEdit();
    if (f.size() < x.size())
        f.resize(x.size());

    if constexpr (nNodes >= 3)
    {
        if (m_gpuRotationMethodSupported)
        {
            gpu::cuda::ElementCorotationalFEMForceFieldCuda_addForceWithRotations<Real, nNodes, dim>(
                nbElem, nbVertex, m_maxElemPerVertex,
                m_gpuElements.deviceRead(), m_gpuInitialRotationsTransposed.deviceRead(),
                m_gpuStiffness.deviceRead(), x.deviceRead(), x0.deviceRead(),
                f.deviceWrite(), m_gpuElementForce.deviceWrite(),
                m_gpuRotations.deviceWrite(), m_gpuVelems.deviceRead());

            m_gpuRotationsUploaded = true;
            d_f.endEdit();
            return;
        }
    }

    // CPU rotations + GPU forces
    this->computeRotations(this->m_rotations, x, x0);
    uploadRotations();

    gpu::cuda::ElementCorotationalFEMForceFieldCuda_addForce<Real, nNodes, dim>(
        nbElem, nbVertex, m_maxElemPerVertex,
        m_gpuElements.deviceRead(), m_gpuRotations.deviceRead(),
        m_gpuStiffness.deviceRead(), x.deviceRead(), x0.deviceRead(),
        f.deviceWrite(), m_gpuElementForce.deviceWrite(),
        m_gpuVelems.deviceRead());

    d_f.endEdit();
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
    constexpr auto nNodes = trait::NumberOfNodesInElement;
    constexpr auto dim = trait::spatial_dimensions;

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    if (df.size() < dx.size())
        df.resize(dx.size());

    const auto kFactor = static_cast<Real>(
        sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
            mparams, this->rayleighStiffness.getValue()));

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto nbElem = static_cast<unsigned int>(elements.size());
    const auto nbVertex = static_cast<unsigned int>(dx.size());

    gpu::cuda::ElementCorotationalFEMForceFieldCuda_addDForce<Real, nNodes, dim>(
        nbElem, nbVertex, m_maxElemPerVertex,
        m_gpuElements.deviceRead(), m_gpuRotations.deviceRead(),
        m_gpuStiffness.deviceRead(), dx.deviceRead(),
        df.deviceWrite(), m_gpuElementForce.deviceWrite(),
        m_gpuVelems.deviceRead(), kFactor);

    d_df.endEdit();
}

} // namespace sofa::component::solidmechanics::fem::elastic
