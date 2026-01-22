#pragma once
#include <sofa/component/solidmechanics/fem/elastic/ElementCorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VecView.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
ElementCorotationalFEMForceField<DataTypes, ElementType>::ElementCorotationalFEMForceField()
    : m_rotationMethods(this)
{
    this->addUpdateCallback("selectRotationMethod", {&this->m_rotationMethods.d_rotationMethod},
        [this](const sofa::core::DataTracker&)
        {
            m_rotationMethods.selectRotationMethod();
            computeInitialRotations();
            return this->getComponentState();
        },
        {});

    m_rotationMethods.selectRotationMethod();
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::init()
{
    BaseElementLinearFEMForceField<DataTypes, ElementType>::init();
    FEMForceField<DataTypes, ElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        m_rotationMethods.selectRotationMethod();
        computeInitialRotations();
    }

    if (!this->isComponentStateInvalid())
    {
        const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::beforeElementForce(
    const sofa::core::MechanicalParams* mparams, sofa::type::vector<ElementForce>& f,
    const sofa::VecCoord_t<DataTypes>& x)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    m_rotations.resize(elements.size(), RotationMatrix::Identity());
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::computeElementsForces(
    const sofa::simulation::Range<std::size_t>& range, const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementForce>& elementForces, const sofa::VecCoord_t<DataTypes>& nodePositions)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto restPositionAccessor = this->mstate->readRestPositions();
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];

        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, nodePositions);
        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> restElementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, restPositionAccessor.ref());

        auto& elementInitialRotationTransposed = this->m_initialRotationsTransposed[elementId];
        auto& elementRotation = this->m_rotations[elementId];

        m_rotationMethods.computeRotation(elementRotation, elementInitialRotationTransposed, elementNodesCoordinates, restElementNodesCoordinates);

        const auto t = translation(elementNodesCoordinates);
        const auto t0 = translation(restElementNodesCoordinates);

        typename trait::ElementDisplacement displacement(sofa::type::NOINIT);
        for (sofa::Size j = 0; j < trait::NumberOfNodesInElement; ++j)
        {
            VecView<trait::spatial_dimensions, sofa::Real_t<DataTypes>> transformedDisplacement(displacement, j * trait::spatial_dimensions);
            transformedDisplacement = elementRotation.multTranspose(elementNodesCoordinates[j] - t) - (restElementNodesCoordinates[j] - t0);
        }

        const auto& stiffnessMatrix = elementStiffness[elementId];

        auto& elementForce = elementForces[elementId];
        elementForce = stiffnessMatrix * displacement;

        for (sofa::Size i = 0; i < trait::NumberOfNodesInElement; ++i)
        {
            VecView<trait::spatial_dimensions, sofa::Real_t<DataTypes>> nodeForce(elementForce, i * trait::spatial_dimensions);
            nodeForce = elementRotation * nodeForce;
        }
    }
}


template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::computeElementsForcesDeriv(
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

        const auto& elementRotation = m_rotations[elementId];

        sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<DataTypes>> element_dx(sofa::type::NOINIT);

        for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
        {
            VecView<trait::spatial_dimensions, sofa::Real_t<DataTypes>> rotated_dx(element_dx, n * trait::spatial_dimensions);
            rotated_dx = elementRotation.multTranspose(nodeDx[element[n]]);
        }

        const auto& stiffnessMatrix = elementStiffness[elementId];

        auto& df = elementForcesDeriv[elementId];
        df = stiffnessMatrix * element_dx;

        for (sofa::Size n = 0; n < trait::NumberOfNodesInElement; ++n)
        {
            VecView<trait::spatial_dimensions, sofa::Real_t<DataTypes>> nodedForce(df, n * trait::spatial_dimensions);
            nodedForce = elementRotation * nodedForce;
        }
    }
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->sofa::core::behavior::ForceField<DataTypes>::mstate)
        .withRespectToPositionsIn(this->sofa::core::behavior::ForceField<DataTypes>::mstate);

    sofa::type::Mat<trait::spatial_dimensions, trait::spatial_dimensions, sofa::Real_t<DataTypes>> localMatrix(sofa::type::NOINIT);

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffness = sofa::helper::getReadAccessor(this->d_elementStiffness);

    if (m_rotations.size() < elements.size() || elementStiffness.size() < elements.size())
    {
        return;
    }

    auto elementStiffnessIt = elementStiffness.begin();
    auto rotationMatrixIt = m_rotations.begin();
    for (const auto& element : elements)
    {
        const auto& elementRotation = *rotationMatrixIt++;
        const auto elementRotation_T = elementRotation.transposed();

        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < trait::NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < trait::NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getAssembledMatrix().getsub(trait::spatial_dimensions * n1, trait::spatial_dimensions * n2, localMatrix);  // extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(element[n1] * trait::spatial_dimensions, element[n2] * trait::spatial_dimensions) += -elementRotation * localMatrix * elementRotation_T;
            }
        }
    }
}

template <class DataTypes, class ElementType>
SReal ElementCorotationalFEMForceField<DataTypes, ElementType>::getPotentialEnergy(
    const sofa::core::MechanicalParams*,
    const sofa::DataVecCoord_t<DataTypes>& x) const
{
    return 0;
}

template <class DataTypes, class ElementType>
auto ElementCorotationalFEMForceField<DataTypes, ElementType>::translation(
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) const -> sofa::Coord_t<DataTypes>
{
    // return nodes[0];
    return computeCentroid(nodes);
}

template <class DataTypes, class ElementType>
auto ElementCorotationalFEMForceField<DataTypes, ElementType>::computeCentroid(
    const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement>& nodes) -> sofa::Coord_t<DataTypes>
{
    sofa::Coord_t<DataTypes> centroid;
    for (const auto node : nodes)
    {
        centroid += node;
    }
    centroid /= static_cast<sofa::Real_t<DataTypes>>(trait::NumberOfNodesInElement);
    return centroid;
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::computeRotations(
    sofa::type::vector<RotationMatrix>& rotations,
    const sofa::VecCoord_t<DataTypes>& nodePositions,
    const sofa::VecCoord_t<DataTypes>& nodeRestPositions)
{
    if (!this->l_topology)
        return;

    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);
    std::ranges::iota_view indices {static_cast<decltype(elements.size())>(0ul), elements.size()};

    rotations.resize(elements.size(), RotationMatrix::Identity());
    if (m_initialRotationsTransposed.size() < elements.size())
    {
        m_initialRotationsTransposed.resize(elements.size(), RotationMatrix::Identity());
    }

    std::for_each(indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];

            const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesCoordinates =
                extractNodesVectorFromGlobalVector(element, nodePositions);
            const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> restElementNodesCoordinates =
                extractNodesVectorFromGlobalVector(element, nodeRestPositions);

            m_rotationMethods.computeRotation(rotations[elementId], m_initialRotationsTransposed[elementId], elementNodesCoordinates, restElementNodesCoordinates);
        }
    );
}

template <class DataTypes, class ElementType>
void ElementCorotationalFEMForceField<DataTypes, ElementType>::computeInitialRotations()
{
    auto restPositionAccessor = this->sofa::core::behavior::ForceField<DataTypes>::mstate->readRestPositions();
    computeRotations(m_initialRotationsTransposed, restPositionAccessor.ref(), restPositionAccessor.ref());

    for (auto& rotation : m_initialRotationsTransposed)
    {
        rotation.transpose();
    }
}

}  // namespace sofa::component::solidmechanics::fem::elastic
