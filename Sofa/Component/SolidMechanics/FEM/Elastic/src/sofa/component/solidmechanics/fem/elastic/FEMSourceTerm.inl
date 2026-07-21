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
#include <sofa/component/solidmechanics/fem/elastic/FEMSourceTerm.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
FEMSourceTerm<DataTypes, ElementType>::FEMSourceTerm()
    : d_nodalSourceDensity(initData(&d_nodalSourceDensity, "nodalSourceDensity", 
                "Source term (per unit volume) sampled at each node. Interpolated inside the "
                "element with the shape functions and integrated on the reference configuration."))
{
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::init()
{
    sofa::core::behavior::ForceField<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        sofa::core::behavior::TopologyAccessor::init();
    }

    if (!this->isComponentStateInvalid() && this->mstate)
    {
        this->resizeNodalSourceDensity(this->mstate->getSize());
    }

    if (!this->isComponentStateInvalid() && this->l_topology && this->mstate)
    {
        this->assembleGlobalMatrix();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::resizeNodalSourceDensity(const std::size_t size)
{
    sofa::helper::WriteAccessor nodalSourceDensity = sofa::helper::getWriteAccessor(d_nodalSourceDensity);

    if (nodalSourceDensity.size() < size)
    {
        nodalSourceDensity.resize(size, sofa::Deriv_t<DataTypes>{});
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::assembleGlobalMatrix()
{
    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    sofa::type::vector<ElementMatrix> elementMatrices;

    // 1. compute the geometry-only matrix of each element
    calculateElementMatrix(elements, elementMatrices);

    // 2. scatter the element matrices into the global matrix
    initializeGlobalMatrix(elements, elementMatrices);
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::calculateElementMatrix(
    const auto& elements, sofa::type::vector<ElementMatrix>& elementMatrices)
{
    const auto restPositionsAccessor = this->mstate->readRestPositions();
    elementMatrices.resize(elements.size());

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        auto& elementMatrix = elementMatrices[elementId];

        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> elementNodesRestCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionsAccessor.ref());

        // M_ij = integral of N_i N_j dV, evaluated on the rest configuration (geometry only).
        for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
        {
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            const auto jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                elementNodesRestCoordinates, dN_dq_ref);
            const auto detJ = sofa::type::absGeneralizedDeterminant(jacobian);

            const auto NT_N = sofa::type::dyad(N, N);

            elementMatrix += (weight * detJ) * NT_N;
        }
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::initializeGlobalMatrix(
    const auto& elements, const sofa::type::vector<ElementMatrix>& elementMatrices)
{
    m_globalMatrix.clear();
    const auto size = this->mstate->getSize();
    m_globalMatrix.resize(size, size);

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        const auto& elementMatrix = elementMatrices[elementId];

        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
            {
                m_globalMatrix.add(element[i], element[j], elementMatrix(i, j));
            }
        }
    }

    m_globalMatrix.compress();
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::addForce(const sofa::core::MechanicalParams* mparams,
                                                     sofa::DataVecDeriv_t<DataTypes>& f,
                                                     const sofa::DataVecCoord_t<DataTypes>& x,
                                                     const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);

    const sofa::helper::ReadAccessor nodalSourceDensity = sofa::helper::getReadAccessor(d_nodalSourceDensity);
    auto forceAccessor = sofa::helper::getWriteAccessor(f);

    // f_i = sum_j M_ij b_j : apply the global matrix to the nodal source density.
    for (std::size_t xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrix::Range rowRange(m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);
        for (typename GlobalMatrix::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            const auto& value = m_globalMatrix.colsValue[xj];

            forceAccessor[rowId] += nodalSourceDensity[columnId] * value;
        }
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::addDForce(const sofa::core::MechanicalParams* mparams,
                                                      sofa::DataVecDeriv_t<DataTypes>& df,
                                                      const sofa::DataVecDeriv_t<DataTypes>& dx)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(df);
    SOFA_UNUSED(dx);
}

template <class DataTypes, class ElementType>
void FEMSourceTerm<DataTypes, ElementType>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    SOFA_UNUSED(matrix);
}

template <class DataTypes, class ElementType>
SReal FEMSourceTerm<DataTypes, ElementType>::getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
                                                                const sofa::DataVecCoord_t<DataTypes>& x) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    return 0.0;
}

}  // namespace sofa::component::solidmechanics::fem::elastic
