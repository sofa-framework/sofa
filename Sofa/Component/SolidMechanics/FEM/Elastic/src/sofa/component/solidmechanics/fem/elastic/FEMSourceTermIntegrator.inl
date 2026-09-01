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
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
FEMSourceTermIntegrator<DataTypes, ElementType>::FEMSourceTermIntegrator()
    : l_constantSources(initLink("constantSources", "Source terms of the weak form integrated by "
                "this component. If empty, the ones found in the current context are used."))
    , l_nonConstantSources(initLink("nonConstantSources", "Displacement-dependent source terms "
                "linked to this component. If empty, the ones found in the current context are "
                "used."))
    , d_quadratureDegree(initData(&d_quadratureDegree, static_cast<sofa::Size>(1), "quadratureDegree",
                "Degree of the quadrature rule integrating the element matrix M."))
    , d_useTangentStiffness(initData(&d_useTangentStiffness, true, "useTangentStiffness",
                "Whether to assemble/apply the stiffness of the displacement-dependent terms."))
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
    // Gather all matching source components in Context if a link is left empty
    auto fallbackToContext = [this](auto& link)
    {
        using SourceType = typename std::remove_reference_t<decltype(link)>::DestType;

        if (link.empty())
        {
            const auto sourcesInContext = this->getContext()->template getObjects<SourceType>(
                sofa::core::objectmodel::BaseContext::Local);

            for (const auto& source : sourcesInContext)
                link.add(source);

            msg_info_when(!sourcesInContext.empty(), this) << "No source term linked: the "
                << sourcesInContext.size() << " one(s) found in the current context are used.";
        }
    };

    fallbackToContext(l_constantSources);
    fallbackToContext(l_nonConstantSources);

    msg_warning_when(l_constantSources.empty() && l_nonConstantSources.empty(), this)
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
    const auto quadraturePointsPerElement = quadratureRule.size();

    m_referenceJacobian.resize(elements.size() * quadraturePointsPerElement);

    for (sofa::Index elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];
        auto& elementMatrix = elementMatrices[elementId];

        const std::array<Coord, NumberOfNodesInElement> elementNodesRestCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionsAccessor.ref());

        // M_ij = integral of N_i N_j dV, evaluated on the rest configuration (geometry only).
        sofa::Index quadraturePointIndex = 0;
        for (const auto& [quadraturePoint, weight] : quadratureRule)
        {
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            auto& jacobian = m_referenceJacobian[elementId * quadraturePointsPerElement + quadraturePointIndex];
            jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                elementNodesRestCoordinates, dN_dq_ref);
            const auto detJ = sofa::type::absGeneralizedDeterminant(jacobian);

            const auto NT_N = sofa::type::dyad(N, N);

            elementMatrix += (weight * detJ) * NT_N;

            ++quadraturePointIndex;
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
    const VecDeriv& nodalSourceTerm, VecDeriv& result) const
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
    VecDeriv sourceTerms(size, Deriv{});

    for (const auto& source : l_constantSources)
    {
        for (sofa::Index i = 0; i < size; ++i)
            sourceTerms[i] += source->getNodeProperty(i);
    }

    m_constantForce.assign(size, Deriv{});
    applyGlobalMatrix(sourceTerms, m_constantForce);
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::addForce(const sofa::core::MechanicalParams* mparams,
                                                     DataVecDeriv& f,
                                                     const DataVecCoord& x,
                                                     const DataVecDeriv& v)
{
    SOFA_UNUSED(mparams);
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

    if (!l_nonConstantSources.empty())
    {
        const sofa::helper::ReadAccessor positionAccessor = sofa::helper::getReadAccessor(x);
        VecDeriv& nonConstantForce = forceAccessor.wref();

        const auto restPositionsAccessor = this->mstate->readRestPositions();
        const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
        const auto quadratureRule = FiniteElement::quadratureRule(d_quadratureDegree.getValue());

        for (const auto& element : elements)
        {
            const std::array<Coord, NumberOfNodesInElement> elementNodesRestCoordinates =
                extractNodesVectorFromGlobalVector(element, restPositionsAccessor.ref());
            const std::array<Coord, NumberOfNodesInElement> elementNodesCoordinates =
                extractNodesVectorFromGlobalVector(element, positionAccessor.ref());

            std::array<Deriv, NumberOfNodesInElement> elementNodesDisplacement;
            for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            {
                elementNodesDisplacement[i] = elementNodesCoordinates[i] - elementNodesRestCoordinates[i];
            }

            for (const auto& [quadraturePoint, weight] : quadratureRule)
            {
                const auto N = FiniteElement::shapeFunctions(quadraturePoint);
                const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

                const auto jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                    elementNodesCoordinates, dN_dq_ref);
                const auto weightTimesDetJ = static_cast<Real>(weight * sofa::type::absGeneralizedDeterminant(jacobian));

                const auto restPosition = FiniteElement::Helper::evaluateValueInElement(elementNodesRestCoordinates, N);
                const auto displacement = FiniteElement::Helper::evaluateValueInElement(elementNodesDisplacement, N);

                for (const auto& source : l_nonConstantSources)
                {
                    const auto sourceDensity = source->evaluate(restPosition, displacement, jacobian);

                    for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                    {
                        nonConstantForce[element[i]] += sourceDensity * (weightTimesDetJ * N[i]);
                    }
                }
            }
        }
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::addDForce(const sofa::core::MechanicalParams* mparams,
                                                      DataVecDeriv& df,
                                                      const DataVecDeriv& dx)
{
    if (this->isComponentStateInvalid() || l_nonConstantSources.empty() || !d_useTangentStiffness.getValue())
    {
        return;
    }

    // never mparams->kFactor() directly, so that Rayleigh stiffness damping is folded in
    const auto kFactor = static_cast<Real>(sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
        mparams, this->rayleighStiffness.getValue()));

    auto forceDerivAccessor = sofa::helper::getWriteAccessor(df);
    const sofa::helper::ReadAccessor positionDerivAccessor = sofa::helper::getReadAccessor(dx);
    const auto positionsAccessor = this->mstate->readPositions();

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    const auto quadratureRule = FiniteElement::quadratureRule(d_quadratureDegree.getValue());

    for (const auto& element : elements)
    {
        const std::array<Coord, NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, positionsAccessor.ref());

        for (const auto& [quadraturePoint, weight] : quadratureRule)
        {
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            const auto jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                elementNodesCoordinates, dN_dq_ref);

            Deriv contraction{};
            for (const auto& source : l_nonConstantSources)
            {
                for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                {
                    contraction += source->evaluateStiffness(jacobian, dN_dq_ref[j]) * positionDerivAccessor[element[j]];
                }
            }

            for (sofa::Size a = 0; a < NumberOfNodesInElement; ++a)
            {
                forceDerivAccessor[element[a]] += contraction * (kFactor * static_cast<Real>(weight) * N[a]);
            }
        }
    }
}

template <class DataTypes, class ElementType>
void FEMSourceTermIntegrator<DataTypes, ElementType>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (this->isComponentStateInvalid() || l_nonConstantSources.empty() || !d_useTangentStiffness.getValue())
    {
        return;
    }

    auto dfdx = matrix->getForceDerivativeIn(this->mstate).withRespectToPositionsIn(this->mstate);

    const auto positionsAccessor = this->mstate->readPositions();
    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    const auto quadratureRule = FiniteElement::quadratureRule(d_quadratureDegree.getValue());

    for (const auto& element : elements)
    {
        const std::array<Coord, NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, positionsAccessor.ref());

        for (const auto& [quadraturePoint, weight] : quadratureRule)
        {
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            const auto jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                elementNodesCoordinates, dN_dq_ref);

            std::array<SourceDerivative, NumberOfNodesInElement> D{};
            for (const auto& source : l_nonConstantSources)
            {
                for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                {
                    D[j] += source->evaluateStiffness(jacobian, dN_dq_ref[j]);
                }
            }

            for (sofa::Size a = 0; a < NumberOfNodesInElement; ++a)
            {
                for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                {
                    dfdx(element[a] * spatial_dimensions, element[j] * spatial_dimensions)
                        += (static_cast<Real>(weight) * N[a]) * D[j];
                }
            }
        }
    }
}

template <class DataTypes, class ElementType>
SReal FEMSourceTermIntegrator<DataTypes, ElementType>::getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
                                                                const DataVecCoord& x) const
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
