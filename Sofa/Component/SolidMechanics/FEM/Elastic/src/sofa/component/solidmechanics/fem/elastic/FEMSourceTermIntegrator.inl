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
                "Degree of the quadrature rule integrating the source terms."))
{
    this->addUpdateCallback("reassembleConstantForce", {&d_quadratureDegree},
        [this](const sofa::core::DataTracker&)
        {
            if (!this->isComponentStateInvalid() && this->l_topology && this->mstate)
            {
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
    // Gather all BaseSourceTerm components in Context if empty
    if (l_constantSources.empty())
    {
        const auto sourcesInContext = this->getContext()->template getObjects<BaseSourceTerm<DataTypes, ElementType> >(
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
void FEMSourceTermIntegrator<DataTypes, ElementType>::assembleConstantForce()
{
    m_constantForce.assign(this->mstate->getSize(), sofa::Deriv_t<DataTypes>{});

    const auto restPositionsAccessor = this->mstate->readRestPositions();
    const auto positionsAccessor = this->mstate->readPositions();

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    const auto quadratureRule = FiniteElement::quadratureRule(d_quadratureDegree.getValue());

    for (const auto& element : elements)
    {
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> elementNodesRestCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionsAccessor.ref());
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, positionsAccessor.ref());

        std::array<sofa::Deriv_t<DataTypes>, NumberOfNodesInElement> elementNodesDisplacement;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            elementNodesDisplacement[i] = elementNodesCoordinates[i] - elementNodesRestCoordinates[i];
        }

        for (const auto& [quadraturePoint, weight] : quadratureRule)
        {
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            const auto jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                elementNodesRestCoordinates, dN_dq_ref);
            const auto measure = static_cast<Real>(sofa::type::absGeneralizedDeterminant(jacobian));

            const auto restPosition =
                FiniteElement::Helper::evaluateValueInElement(elementNodesRestCoordinates, N);
            const auto displacement =
                FiniteElement::Helper::evaluateValueInElement(elementNodesDisplacement, N);

            const QuadratureContext<DataTypes, ElementType> context{
                element, N, dN_dq_ref, jacobian, measure, restPosition, displacement};

            const auto weightTimesMeasure = static_cast<Real>(weight) * measure;

            for (const auto& source : l_constantSources)
            {
                const auto density = source->evaluate(context);

                for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                {
                    m_constantForce[element[i]] += density * (weightTimesMeasure * N[i]);
                }
            }
        }
    }
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
