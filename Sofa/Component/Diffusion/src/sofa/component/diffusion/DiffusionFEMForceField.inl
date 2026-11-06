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

#include <sofa/component/diffusion/DiffusionFEMForceField.h>

#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <algorithm>

namespace sofa::component::diffusion
{
template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::DiffusionFEMForceField()
    : d_elementDiffusionCoefficient(initData(
          &d_elementDiffusionCoefficient,
          sofa::type::vector<Real>(defaultDiffusionCoefficient),
          "elementDiffusionCoefficient",
          "List of isotropic diffusion coefficient per element. The last coefficient is used if the list does not match the number of elements."))
    , d_elementHessian(initData(
          &d_elementHessian,
          "elementStiffness",
          "Precomputed diffusion stiffness matrix per element"))
    , l_spatialState(initLink("spatialState", "Link to the state holding the spatial positions of the mesh"))
{
    this->addUpdateCallback("precomputeDiffusionStiffness", {&d_elementDiffusionCoefficient},
                            [this](const sofa::core::DataTracker&)
                            {
                                precomputeElementStiffness();
                                return this->getComponentState();
                            },
                            {});
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::validateSpatialState()
{
    if (!l_spatialState.get())
    {
        l_spatialState.set(
            this->getContext()->template get<sofa::core::behavior::MechanicalState<SpatialTypes> >());

        if (!l_spatialState)
        {
            msg_error() << "No compatible MechanicalState<SpatialTypes> found in the current context. "
                           "Please set the link `spatialState` explicitly.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
        else
        {
            msg_info() << "No link given for `spatialState`, but it has been found in the context";
        }
    }
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::init()
{
    Inherited::init();

    if (!this->isComponentStateInvalid())
    {
        validateSpatialState();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        precomputeElementStiffness();
    }
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::precomputeElementStiffness()
{
    if (!this->l_topology || !this->mstate || !this->l_spatialState)
        return;

    if (this->isComponentStateInvalid())
        return;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

    const auto elementDiffusion = sofa::helper::getReadAccessor(d_elementDiffusionCoefficient);

    const auto restPositionAccessor = this->l_spatialState->readRestPositions();

    if (restPositionAccessor.size() != this->mstate->getSize())
    {
        msg_error() << "The spatial state (" << restPositionAccessor.size()
                    << " nodes) and the diffused field state (" << this->mstate->getSize()
                    << " nodes) must have the same number of nodes";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    auto elementHessian = sofa::helper::getWriteOnlyAccessor(d_elementHessian);
    elementHessian.resize(elements.size());

    SCOPED_TIMER("precomputeDiffusionHessian");

    sofa::helper::IotaView indices { static_cast<decltype(elements.size())>(0ul), elements.size() };
    std::for_each(indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];
            const Real diffusion = getElementDiffusionCoefficient(elementDiffusion, elementId);

            const std::array<GeometryCoord, NumberOfNodes> nodesCoordinates =
                sofa::component::solidmechanics::fem::elastic::extractNodesVectorFromGlobalVector(
                    element, restPositionAccessor.ref());

            ElementHessian K;
            K.clear();

            for (const auto& [quadraturePoint, weight] : trait::FiniteElement::quadraturePoints())
            {
                // Gradients of the shape functions in the reference element
                const sofa::type::Mat<NumberOfNodes, trait::TopologicalDimension, GeometryReal> dN_dq_ref =
                    trait::FiniteElement::gradientShapeFunctions(quadraturePoint);

                // Jacobian of the geometric mapping (spatial dim x topological dim): geometry only
                sofa::type::Mat<SpatialDimension, trait::TopologicalDimension, GeometryReal> jacobian;
                jacobian.clear();
                for (sofa::Size i = 0; i < NumberOfNodes; ++i)
                {
                    jacobian += sofa::type::dyad(nodesCoordinates[i], dN_dq_ref[i]);
                }

                const auto detJ = sofa::type::absGeneralizedDeterminant(jacobian);

                const sofa::type::Mat<trait::TopologicalDimension, SpatialDimension, GeometryReal> J_inv =
                    sofa::type::inverse(jacobian);

                // Gradients of the shape functions in the spatial frame
                sofa::type::Mat<NumberOfNodes, SpatialDimension, GeometryReal> dN_dx(sofa::type::NOINIT);
                for (sofa::Size i = 0; i < NumberOfNodes; ++i)
                {
                    dN_dx[i] = J_inv.transposed() * dN_dq_ref[i];
                }

                const Real integrationFactor = diffusion * static_cast<Real>(weight * detJ);

                for (sofa::Index i = 0; i < NumberOfNodes; ++i)
                {
                    for (sofa::Index j = 0; j < NumberOfNodes; ++j)
                    {
                        // scalar product of the spatial gradients: sum over SPATIAL dimensions
                        GeometryReal gradDot {};
                        for (sofa::Index d = 0; d < SpatialDimension; ++d)
                        {
                            gradDot += dN_dx[i][d] * dN_dx[j][d];
                        }

                        const Real scalarStiffness = integrationFactor * static_cast<Real>(gradDot);

                        // the same scalar stiffness applies to each component of the FIELD
                        for (sofa::Index c = 0; c < FieldDimension; ++c)
                        {
                            K(i * FieldDimension + c, j * FieldDimension + c) += scalarStiffness;
                        }
                    }
                }
            }

            elementHessian[elementId] = K;
        });
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
template <class VecType>
auto DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::computeElementValue(
    const typename trait::TopologyElement& element,
    const VecType& nodeValues) const -> ElementValue
{
    ElementValue values { sofa::type::NOINIT };

    for (sofa::Size nodeId = 0; nodeId < NumberOfNodes; ++nodeId)
    {
        const auto globalNodeId = element[nodeId];

        for (sofa::Size c = 0; c < FieldDimension; ++c)
        {
            values[nodeId * FieldDimension + c] = nodeValues[globalNodeId][c];
        }
    }

    return values;
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
auto DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::getElementDiffusionCoefficient(
        const sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<Real>>>& elementDiffusionCoefficient,
        sofa::Index elementId) const -> Real
{
    if (elementId < elementDiffusionCoefficient.size())
        return elementDiffusionCoefficient[elementId];
    if (!elementDiffusionCoefficient.empty())
        return elementDiffusionCoefficient->back();
    return defaultDiffusionCoefficient;
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::computeElementsForces(
    const sofa::simulation::Range<std::size_t>& range,
    const sofa::core::MechanicalParams*,
    sofa::type::vector<ElementGradient>& elementForces,
    const sofa::VecCoord_t<DiffusionFieldTypes>& nodeValues)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(d_elementHessian);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto elementValue = computeElementValue(elements[elementId], nodeValues);
        elementForces[elementId] = (elementStiffness[elementId] * elementValue);
    }
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::computeElementsForcesDeriv(
    const sofa::simulation::Range<std::size_t>& range,
    const sofa::core::MechanicalParams*,
    sofa::type::vector<ElementGradient>& elementForcesDeriv,
    const sofa::VecDeriv_t<DiffusionFieldTypes>& nodeDx)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(d_elementHessian);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto elementDx = computeElementValue(elements[elementId], nodeDx);

        elementForcesDeriv[elementId] = (elementStiffness[elementId] * elementDx);
    }
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (this->isComponentStateInvalid() || !this->l_topology)
        return;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
        .withRespectToPositionsIn(this->mstate);

    FieldBlock localMatrix { sofa::type::NOINIT };

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(d_elementHessian);

    if (elementStiffness.size() < elements.size())
        return;

    auto elementStiffnessIt = elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < NumberOfNodes; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < NumberOfNodes; ++n2)
            {
                stiffnessMatrix.getsub(
                    FieldDimension * n1,
                    FieldDimension * n2,
                    localMatrix);

                dfdx(
                    element[n1] * FieldDimension,
                    element[n2] * FieldDimension) += -localMatrix;
            }
        }
    }
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
SReal DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::getPotentialEnergy(
    const sofa::core::MechanicalParams*,
    const sofa::DataVecCoord_t<DiffusionFieldTypes>& fieldValues) const
{
    if (this->isComponentStateInvalid() || !this->l_topology)
        return 0;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(d_elementHessian);
    const auto fieldAccessor = sofa::helper::getReadAccessor(fieldValues);

    if (elementStiffness.size() < elements.size())
        return 0;

    Real energy {};

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto elementValue = computeElementValue(elements[elementId], fieldAccessor.ref());

        energy += elementValue * (elementStiffness[elementId] * elementValue);
    }

    return static_cast<SReal>(0.5 * energy);
}

template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
void DiffusionFEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>::addKToMatrix(
    sofa::linearalgebra::BaseMatrix* matrix,
    SReal kFact,
    unsigned& offset)
{
    if (this->isComponentStateInvalid() || !this->l_topology)
        return;

    FieldBlock localMatrix { sofa::type::NOINIT };

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    const auto elementStiffness = sofa::helper::getReadAccessor(d_elementHessian);

    if (elementStiffness.size() < elements.size())
        return;

    auto elementStiffnessIt = elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < NumberOfNodes; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < NumberOfNodes; ++n2)
            {
                stiffnessMatrix.getsub(
                    FieldDimension * n1,
                    FieldDimension * n2,
                    localMatrix);

                const auto value =
                    (-static_cast<Real>(kFact))
                    * static_cast<sofa::type::ScalarOrMatrix<FieldBlock> >(localMatrix);

                matrix->add(
                    offset + element[n1] * FieldDimension,
                    offset + element[n2] * FieldDimension,
                    value);
            }
        }
    }
}

} // namespace sofa::component::diffusion
