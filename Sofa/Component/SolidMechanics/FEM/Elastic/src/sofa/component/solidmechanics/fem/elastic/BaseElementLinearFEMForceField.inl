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

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/impl/LameParameters.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>

#include <ranges>

#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
BaseElementLinearFEMForceField<DataTypes, ElementType>::BaseElementLinearFEMForceField()
    : d_elementStiffness(initData(&d_elementStiffness, "elementStiffness", "List of stiffness matrices per element"))
{
    this->addUpdateCallback("precomputeStiffness", {&this->d_youngModulus, &this->d_poissonRatio},
    [this](const sofa::core::DataTracker& )
    {
        precomputeElementStiffness();
        return this->getComponentState();
    }, {});
}

template <class DataTypes, class ElementType>
void BaseElementLinearFEMForceField<DataTypes, ElementType>::init()
{
    sofa::component::solidmechanics::fem::elastic::BaseLinearElasticityFEMForceField<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        this->precomputeElementStiffness();
    }
}

template <class DataTypes, class ElementType>
void BaseElementLinearFEMForceField<DataTypes, ElementType>::precomputeElementStiffness()
{
    if (!this->l_topology)
        return;

    if (this->isComponentStateInvalid())
        return;

    if (!this->mstate)
        return;

    const auto youngModulusAccessor = sofa::helper::ReadAccessor(this->d_youngModulus);
    const auto poissonRatioAccessor = sofa::helper::ReadAccessor(this->d_poissonRatio);

    auto restPositionAccessor = this->mstate->readRestPositions();

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

    auto elementStiffness = sofa::helper::getWriteOnlyAccessor(d_elementStiffness);
    elementStiffness.resize(elements.size());

    SCOPED_TIMER("precomputeStiffness");
    sofa::helper::IotaView indices {static_cast<decltype(elements.size())>(0ul), elements.size()};
    std::for_each(indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];

            const auto youngModulus = this->getYoungModulusInElement(elementId);
            const auto poissonRatio = this->getPoissonRatioInElement(elementId);

            const auto [mu, lambda] = sofa::component::solidmechanics::fem::elastic::toLameParameters<DataTypes>(youngModulus, poissonRatio);

            const auto elasticityTensor = makeIsotropicElasticityTensor<DataTypes>(mu, lambda);

            const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> nodesCoordinates = extractNodesVectorFromGlobalVector(element, restPositionAccessor.ref());
            elementStiffness[elementId] = integrate<DataTypes, ElementType, trait::matrixVectorProductType>(nodesCoordinates, elasticityTensor);
        });
}

}
