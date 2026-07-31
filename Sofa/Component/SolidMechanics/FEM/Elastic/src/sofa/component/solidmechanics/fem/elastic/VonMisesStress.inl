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
#include <sofa/component/solidmechanics/fem/elastic/VonMisesStress.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/DrawColoredMesh.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/IotaView.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
VonMisesStress<DataTypes, ElementType>::VonMisesStress()
    : d_nodalStress(initData(&d_nodalStress, sofa::type::vector<LocalStressValues>{}, "nodalStress",
                             "Local nodal von Mises stress values"))
    , d_continuousField(initData(&d_continuousField, false, "continuousField",
    "Compute von Mises stress as a continuous field across the elements. Necessitate the solve "
        "of a sparse linear system. Otherwise, the von Mises stress is computed locally. A local "
        "stress value may indicate discretization errors if the field does not appear continuous."))
    , d_colorMap(initData(&d_colorMap, sofa::helper::ColorMap(), "colorMap", "Color map"))
    , l_stressEvaluator(initLink("stressEvaluator", "The component in charge of evaluating the Cauchy stress."))
{
    // This component must receive events
    f_listening.setValue(true);
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::init()
{
    core::behavior::SingleStateAccessor<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateTopology();
    }

    if (!this->isComponentStateInvalid())
    {
        validateStressEvaluatorLink();
    }

    if (!this->isComponentStateInvalid())
    {
        auto nodalStress = sofa::helper::getWriteOnlyAccessor(d_nodalStress);
        const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
        nodalStress->resize(elements.size());

        this->precomputeData();
        this->calculateElementMassMatrix(elements, m_elementMassMatrices);
    }
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::handleEvent(core::objectmodel::Event* event)
{
    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
        const auto nbElements = elements.size();

        auto nodalStress = sofa::helper::getWriteOnlyAccessor(d_nodalStress);
        nodalStress->clear();
        nodalStress->resize(nbElements);

        auto positionAccessor = this->mstate->readPositions();

        helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};

        std::for_each(
            indices.begin(), indices.end(),
            [&](const auto elementId)
            {
                const auto& element = elements[elementId];

                std::array<Coord_t<DataTypes>, NumberOfNodesInElement> nodeCoordinatesInElement;
                for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                {
                    nodeCoordinatesInElement[i] = positionAccessor[element[i]];
                }

                std::array<StressVoigtVector, NumberOfNodesInElement> nodalStressInElement;
                static constexpr auto gradients = sofa::fem::FiniteElementHelper<ElementType, DataTypes>::gradientShapeFunctionAtQuadraturePoints();
                static constexpr auto quadraturePoints = FiniteElement::quadraturePoints();

                std::array<sofa::type::Vec<NumberOfNodesInElement, sofa::Real_t<DataTypes>>, sofa::type::NumberOfIndependentElements<spatial_dimensions>> b;
                for (auto& vec : b) vec.clear();

                for (sofa::Size q = 0; q < NumberOfQuadraturePoints; ++q)
                {
                    const auto& weight = quadraturePoints[q].second;
                    const auto& precomputed = m_precomputedData[elementId][q];

                    // gradient of shape functions in the reference element evaluated at the quadrature point
                    const auto& dN_dq_ref = gradients[q];

                    // jacobian of the mapping from the reference space to the CURRENT physical space
                    const auto J_q = FiniteElement::Helper::jacobianFromReferenceToPhysical(nodeCoordinatesInElement, dN_dq_ref);

                    // Deformation Gradient F = J_curr * J_rest_inv
                    const DeformationGradient F = J_q * precomputed.jacobianInv;

                    const auto detJ = sofa::type::absGeneralizedDeterminant(J_q);

                    // shape functions in the reference element evaluated at the quadrature point
                    const auto N = FiniteElement::shapeFunctions(quadraturePoints[q].first);

                    const StressVoigtVector stress = l_stressEvaluator->computeStress(F, elementId);

                    const auto commonFactor = weight * detJ;
                    for (sofa::Size i = 0; i < sofa::type::NumberOfIndependentElements<spatial_dimensions>; ++i)
                    {
                        const auto stressFactor = commonFactor * stress[i];
                        for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                        {
                            b[i][j] += N[j] * stressFactor;
                        }
                    }
                }

                for (sofa::Size i = 0; i < sofa::type::NumberOfIndependentElements<spatial_dimensions>; ++i)
                {
                    const auto stressCoordinate = m_elementMassMatrices[elementId] * b[i];
                    for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                    {
                        nodalStressInElement[j][i] = stressCoordinate[j];
                    }
                }

                for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                {
                    nodalStress[elementId][i] = vonMisesStress(deviatoricStress(nodalStressInElement[i]));
                }

            });
    }
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::validateStressEvaluatorLink()
{
    if (l_stressEvaluator.empty())
    {
        msg_info() << "Link to a valid stress evaluator should be set to ensure right behavior. The first "
                      "stress evaluator found in current context will be used.";
        l_stressEvaluator.set(this->getContext()->template get<CauchyStressEvaluator<DataTypes>>());
    }

    if (l_stressEvaluator == nullptr)
    {
        msg_error() << "No stress evaluator component found at path: '" << this->l_stressEvaluator.getLinkedPath()
                    << "', nor in current context: " << this->getContext()->name
                    << ". Object must have a stress evaluator. "
                    << "The list of available stress evaluator components is: "
                    << sofa::core::ObjectFactory::getInstance()
                           ->listClassesDerivedFrom<CauchyStressEvaluator<DataTypes>>();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::precomputeData()
{
    if (this->l_topology == nullptr) return;

    auto restPositionAccessor = this->mstate->readRestPositions();
    const auto& restPosition = restPositionAccessor.ref();

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    m_precomputedData.resize(elements.size());

    static constexpr auto gradients = sofa::fem::FiniteElementHelper<ElementType, DataTypes>::gradientShapeFunctionAtQuadraturePoints();

    for (std::size_t i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        std::array<Coord_t<DataTypes>, NumberOfNodesInElement> nodeCoordinatesInElement;
        for (sofa::Size n = 0; n < NumberOfNodesInElement; ++n)
            nodeCoordinatesInElement[n] = restPosition[element[n]];

        for (std::size_t j = 0; j < NumberOfQuadraturePoints; ++j)
        {
            const auto& dN_dq_ref = gradients[j];
            PrecomputedData& data = m_precomputedData[i][j];
            data.jacobian = sofa::fem::FiniteElementHelper<ElementType, DataTypes>::jacobianFromReferenceToPhysical(nodeCoordinatesInElement, dN_dq_ref);
            data.jacobianInv = sofa::type::inverse(data.jacobian);
            data.detJacobian = sofa::type::absGeneralizedDeterminant(data.jacobian);

            for (sofa::Size n = 0; n < NumberOfNodesInElement; ++n)
            {
                data.dN_dQ[n] = data.jacobianInv.multTranspose(dN_dq_ref[n]);
            }
        }
    }
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::calculateElementMassMatrix(
    const auto& elements, sofa::type::vector<ElementMassMatrix>& elementMassMatrices)
{
    const auto nbElements = elements.size();
    elementMassMatrices.resize(nbElements);

    auto positionAccessor = this->mstate->readPositions();

    SCOPED_TIMER("elementMassMatrix");
    helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};
    std::for_each(
        indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];
            auto& elementMassMatrix = elementMassMatrices[elementId];

            std::array<Coord_t<DataTypes>, NumberOfNodesInElement> nodeCoordinatesInElement;
            for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            {
                nodeCoordinatesInElement[i] = positionAccessor[element[i]];
            }

            for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
            {
                // gradient of shape functions in the reference element evaluated at the quadrature
                // point
                const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real_t<DataTypes>>
                    dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

                // jacobian of the mapping from the reference space to the physical space, evaluated
                // at the quadrature point
                sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real_t<DataTypes>>
                    jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                        nodeCoordinatesInElement, dN_dq_ref);

                const auto detJ = sofa::type::absGeneralizedDeterminant(jacobian);

                // shape functions in the reference element evaluated at the quadrature point
                const auto N = FiniteElement::shapeFunctions(quadraturePoint);

                const auto NT_N = sofa::type::dyad(N, N);

                elementMassMatrix += (weight * detJ) * NT_N;
            }

            sofa::type::invertMatrix(elementMassMatrix, elementMassMatrix);
        });
}

template <class DataTypes, class ElementType>
auto VonMisesStress<DataTypes, ElementType>::deviatoricStress(const StressVoigtVector& sigma) -> StressVoigtVector
{
    StressVoigtVector s = sigma;

    Real_t<DataTypes> trace {};
    for (sofa::Size i = 0; i < spatial_dimensions; ++i)
    {
        trace += sigma[type::tensorToVoigtIndex<spatial_dimensions>(i, i)];
    }

    static constexpr Real_t<DataTypes> dim_inv = static_cast<Real_t<DataTypes>>(1.0 / spatial_dimensions);

    for (sofa::Size i = 0; i < spatial_dimensions; ++i)
    {
        s[type::tensorToVoigtIndex<spatial_dimensions>(i, i)] -= dim_inv * trace;
    }

    return s;
}

template <class DataTypes, class ElementType>
Real_t<DataTypes>
VonMisesStress<DataTypes, ElementType>::vonMisesStress(const StressVoigtVector& deviatoricStress)
{
    sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real_t<DataTypes>> s;
    for (sofa::Size i = 0; i < spatial_dimensions; ++i)
    {
        for (sofa::Size j = 0; j < spatial_dimensions; ++j)
        {
            s[i][j] = deviatoricStress[type::tensorToVoigtIndex<spatial_dimensions>(i, j)];
        }
    }

    return std::sqrt(static_cast<Real_t<DataTypes>>(3) / 2 * sofa::type::trace(s.multTranspose(s)));
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->disableLighting();

    const auto nodalStress = d_nodalStress.getValue();
    if (nodalStress.empty())
        return;

    const auto positions = this->mstate->readPositions();
    const auto& colorMap = d_colorMap.getValue();

    Real_t<DataTypes> minStress = std::numeric_limits<Real_t<DataTypes>>::max();
    Real_t<DataTypes> maxStress = std::numeric_limits<Real_t<DataTypes>>::lowest();

    for (const auto& elementStress : nodalStress)
    {
        for (const auto& stress : elementStress)
        {
            minStress = std::min(minStress, stress);
            maxStress = std::max(maxStress, stress);
        }
    }

    const auto evaluator = colorMap.getEvaluator(minStress, maxStress);

    sofa::type::vector<std::array<type::RGBAColor, NumberOfNodesInElement>> nodesColors;
    for (const auto& elementStress : nodalStress)
    {
        std::array<type::RGBAColor, NumberOfNodesInElement> nodesColorsInElement;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            nodesColorsInElement[i] = (evaluator(elementStress[i]));
        }
        nodesColors.push_back(nodesColorsInElement);
    }

    m_renderer.drawAllElements(vparams->drawTool(), positions.ref(), this->l_topology.get(), nodesColors);
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::computeBBox(const core::ExecParams* params, bool)
{
    if (!this->mstate) return;

    const auto bbox = this->mstate->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
    this->f_bbox.setValue(std::move(bbox));
}

}  // namespace sofa::component::solidmechanics::fem::elastic
