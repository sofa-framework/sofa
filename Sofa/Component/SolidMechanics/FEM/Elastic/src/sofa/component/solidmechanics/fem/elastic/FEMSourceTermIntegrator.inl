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
#include <sofa/component/solidmechanics/fem/elastic/FEMSourceTermIntegrator.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
FEMSourceTermIntegrator<DataTypes, ElementType>::FEMSourceTermIntegrator()
    : l_constantSources(initLink("constantSources", "Source terms of the weak form integrated by "
                "this component. If empty, the ones found in the current context are used."))
    , d_quadratureDegree(initData(&d_quadratureDegree, static_cast<sofa::Size>(1), "quadratureDegree",
                "Degree of the quadrature rule integrating the element matrix M."))
{
    // Re-compute global matrix and constant forces in case of quadrature degree change
    this->addUpdateCallback("reassembleSourceMatrix", {&d_quadratureDegree},
        [this](const sofa::core::DataTracker&)
        {
            if (!this->isComponentStateInvalid() && this->l_topology && this->mstate)
            {
                assembleGlobalMatrix();
                assembleConstantForce();
            }

            return this->getComponentState();
        }, {});
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::init()
{
    sofa::core::behavior::ForceField<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        sofa::core::behavior::TopologyAccessor::init();
    }

    if (!this->isComponentStateInvalid())
    {
        this->validateSources();
    }

    if (!this->isComponentStateInvalid() && this->l_topology && this->mstate)
    {
        this->assembleGlobalMatrix();
        this->assembleConstantForce();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::validateSources()
{
    // Gather all ConstantSourceTerm components in Context if empty
    if (l_constantSources.empty())
    {
        const auto sourcesInContext = this->getContext()->template getObjects<ConstantSourceTerm<DataTypes> >(
            sofa::core::objectmodel::BaseContext::Local);

        for (const auto& source : sourcesInContext)
            l_constantSources.add(source);

        msg_info_when(!sourcesInContext.empty(), this) << "No source term linked: the "
            << sourcesInContext.size() << " one(s) found in the current context are used.";
    }

    msg_warning_when(l_constantSources.empty(), this)
        << "No source term linked, and none found in the current context '"
        << this->getContext()->getName() << "'. This component has zero force contribution.";
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::assembleGlobalMatrix()
{
    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    sofa::type::vector<ElementMatrix> elementMatrices;

    // 1. compute the geometry-only matrix of each element
    calculateElementMatrix(elements, elementMatrices);

    // 2. scatter the element matrices into the global matrix
    initializeGlobalMatrix(elements, elementMatrices);
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::calculateElementMatrix(
    const auto& elements, sofa::type::vector<ElementMatrix>& elementMatrices)
{
    const auto restPositionsAccessor = this->mstate->readRestPositions();
    elementMatrices.resize(elements.size());

    const auto quadratureRule = FiniteElement::quadratureRule(d_quadratureDegree.getValue());

    for (sofa::Index elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        auto& elementMatrix = elementMatrices[elementId];

        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> elementNodesRestCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionsAccessor.ref());

        // M_ij = integral of N_i N_j dV, evaluated on the rest configuration (geometry only).
        for (const auto& [quadraturePoint, weight] : quadratureRule)
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
void FEMSourceTermIntegrator<DataTypes, ElementType>::initializeGlobalMatrix(
    const auto& elements, const sofa::type::vector<ElementMatrix>& elementMatrices)
{
    m_globalMatrix.clear();
    const auto size = this->mstate->getSize();
    m_globalMatrix.resize(size, size);

    for (sofa::Index elementId = 0; elementId < elements.size(); ++elementId)
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
void FEMSourceTermIntegrator<DataTypes, ElementType>::applyGlobalMatrix(
    const sofa::VecDeriv_t<DataTypes>& nodalSourceTerm, sofa::VecDeriv_t<DataTypes>& result) const
{
    // f_i = sum_j M_ij b_j : apply the global matrix to the nodal source term.
    for (sofa::Index xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrix::Range rowRange(m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);
        for (typename GlobalMatrix::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            const auto& value = m_globalMatrix.colsValue[xj];

            result[rowId] += nodalSourceTerm[columnId] * value;
        }
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::assembleConstantForce()
{
    const auto size = this->mstate->getSize();

    // Aggregate all contributions to one vector before applying the global matrix
    sofa::VecDeriv_t<DataTypes> sourceTerms(size, sofa::Deriv_t<DataTypes>{});

    for (const auto& source : l_constantSources)
    {
        for (sofa::Index i = 0; i < size; ++i)
            sourceTerms[i] += source->getNodeProperty(i);
    }

    m_constantForce.assign(size, sofa::Deriv_t<DataTypes>{});
    applyGlobalMatrix(sourceTerms, m_constantForce);
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::addForce(const sofa::core::MechanicalParams* mparams,
                                                     sofa::DataVecDeriv_t<DataTypes>& f,
                                                     const sofa::DataVecCoord_t<DataTypes>& x,
                                                     const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);

    if (this->isComponentStateInvalid())
    {
        return;
    }

    auto forceAccessor = sofa::helper::getWriteAccessor(f);

    for (sofa::Index i = 0; i < m_constantForce.size(); ++i)
    {
        forceAccessor[i] += m_constantForce[i];
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::addDForce(const sofa::core::MechanicalParams* mparams,
                                                      sofa::DataVecDeriv_t<DataTypes>& df,
                                                      const sofa::DataVecDeriv_t<DataTypes>& dx)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(df);
    SOFA_UNUSED(dx);
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    SOFA_UNUSED(matrix);
}

template <class DataTypes, class ElementType>
SReal FEMSourceTermIntegrator<DataTypes, ElementType>::getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
                                                                const sofa::DataVecCoord_t<DataTypes>& x) const
{
    SOFA_UNUSED(mparams);

    if (this->isComponentStateInvalid())
    {
        return 0.0;
    }

    const sofa::helper::ReadAccessor positionAccessor = sofa::helper::getReadAccessor(x);
    const auto restPositionAccessor = this->mstate->readRestPositions();

    SReal energy = 0.0;
    for (sofa::Index i = 0; i < m_constantForce.size(); ++i)
    {
        energy -= dot(m_constantForce[i], positionAccessor[i] - restPositionAccessor.ref()[i]);
    }
    return energy;
}

}  // namespace sofa::component::solidmechanics::fem::elastic
