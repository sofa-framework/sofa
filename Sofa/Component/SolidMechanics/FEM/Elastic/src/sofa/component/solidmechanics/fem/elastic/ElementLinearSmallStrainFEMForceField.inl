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
#include <sofa/component/solidmechanics/fem/elastic/ElementLinearSmallStrainFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
void ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::init()
{
    BaseElementLinearFEMForceField<DataTypes, ElementType>::init();
    FEMForceField<DataTypes, ElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}


template <class DataTypes, class ElementType>
void ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::computeElementsForces(
    const sofa::simulation::Range<std::size_t>& range,
    const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementForce>& elementForces,
    const sofa::VecCoord_t<DataTypes>& nodePositions)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto restPositionAccessor = this->mstate->readRestPositions();
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];
        const auto& stiffnessMatrix = elementStiffness[elementId];

        typename trait::ElementDisplacement displacement{ sofa::type::NOINIT };

        for (sofa::Size j = 0; j < trait::NumberOfNodesInElement; ++j)
        {
            const auto nodeId = element[j];
            for (sofa::Size dim = 0; dim < trait::spatial_dimensions; ++dim)
            {
                displacement[j * trait::spatial_dimensions + dim] = nodePositions[nodeId][dim] - restPositionAccessor[nodeId][dim];
            }
        }

        elementForces[elementId] = stiffnessMatrix * displacement;
    }
}

template <class DataTypes, class ElementType>
void ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::computeElementsForcesDeriv(
    const sofa::simulation::Range<std::size_t>& range,
    const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementForce>& elementForcesDeriv,
    const sofa::VecDeriv_t<DataTypes>& nodeDx)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];
        const auto& stiffnessMatrix = elementStiffness[elementId];

        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesDx =
            extractNodesVectorFromGlobalVector(element, nodeDx);

        sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>> element_dx(sofa::type::NOINIT);
        for (sofa::Size nodeId = 0; nodeId < trait::NumberOfNodesInElement; ++nodeId)
        {
            const auto& dx = elementNodesDx[nodeId];
            for (sofa::Size dim = 0; dim < trait::spatial_dimensions; ++dim)
            {
                element_dx[nodeId * trait::spatial_dimensions + dim] = dx[dim];
            }
        }

        elementForcesDeriv[elementId] = stiffnessMatrix * element_dx;
    }
}

template <class DataTypes, class ElementType>
void ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (this->isComponentStateInvalid())
        return;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
        .withRespectToPositionsIn(this->mstate);

    sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>> localMatrix(sofa::type::NOINIT);

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

    const auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    if (elementStiffness.size() < elements.size())
    {
        return;
    }

    auto elementStiffnessIt = elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < trait::NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < trait::NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getAssembledMatrix().getsub(trait::spatial_dimensions * n1, trait::spatial_dimensions * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(element[n1] * trait::spatial_dimensions, element[n2] * trait::spatial_dimensions) += -localMatrix;
            }
        }
    }
}

template <class DataTypes, class ElementType>
SReal ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::getPotentialEnergy(
    const sofa::core::MechanicalParams*,
    const sofa::DataVecCoord_t<DataTypes>& x) const
{
    return 0;
}

template <class DataTypes, class ElementType>
void ElementLinearSmallStrainFEMForceField<DataTypes, ElementType>::addKToMatrix(
    sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned& offset)
{
    if (this->isComponentStateInvalid())
        return;

    using LocalMatType = sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>>;
    LocalMatType localMatrix{sofa::type::NOINIT};

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);
    auto elementStiffnessIt = elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < trait::NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < trait::NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getAssembledMatrix().getsub(trait::spatial_dimensions * n1, trait::spatial_dimensions * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2

                const auto value = (-static_cast<sofa::Real_t<DataTypes>>(kFact)) * static_cast<ScalarOrMatrix<LocalMatType>>(localMatrix);
                matrix->add(
                   offset + element[n1] * trait::spatial_dimensions,
                   offset + element[n2] * trait::spatial_dimensions, value);
            }
        }
    }
}

}  // namespace sofa::component::solidmechanics::fem::elastic
