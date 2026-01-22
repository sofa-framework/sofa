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
    std::ranges::iota_view indices {static_cast<decltype(elements.size())>(0ul), elements.size()};
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
