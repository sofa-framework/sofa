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

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>
#include <sofa/component/solidmechanics/fem/elastic/CauchyStressEvaluator.h>
#include <sofa/core/behavior/SingleStateAccessor.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/core/visual/DrawColoredMesh.h>
#include <sofa/helper/ColorMap.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
class VonMisesStress : public core::behavior::TopologyAccessor, public core::behavior::SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS2(
        SOFA_TEMPLATE2(VonMisesStress, DataTypes, ElementType),
        core::behavior::TopologyAccessor,
        core::behavior::SingleStateAccessor<DataTypes>);

protected:
    using FiniteElement = sofa::fem::FiniteElement<ElementType, DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;
    static constexpr sofa::Size NumberOfQuadraturePoints = FiniteElement::quadraturePoints().size();

    // a stress tensor represented as a vector using the Voigt mapping
    using StressVoigtVector = sofa::type::Vec<sofa::type::NumberOfIndependentElements<spatial_dimensions>, sofa::Real_t<DataTypes>>;

    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real_t<DataTypes>>;

public:

    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    // A stress value for each node in an element
    using LocalStressValues = std::array<sofa::Real_t<DataTypes>, NumberOfNodesInElement>;
    Data<sofa::type::vector<LocalStressValues>> d_nodalStress;

    Data<helper::ColorMap> d_colorMap;
    Data<bool> d_lighting;

    sofa::SingleLink<MyType, CauchyStressEvaluator<DataTypes>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_stressEvaluator;

    VonMisesStress();

    void handleEvent(core::objectmodel::Event*) override;

protected:

    void validateStressEvaluatorLink();

    using ElementGramMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, sofa::Real_t<DataTypes>>;
    sofa::type::vector<ElementGramMatrix> m_elementInverseGramMatrices;

    struct PrecomputedData
    {
        sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real_t<DataTypes>> jacobian { sofa::type::NOINIT };
        sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real_t<DataTypes>> jacobianInv { sofa::type::NOINIT };
        Real_t<DataTypes> detJacobian {};
        sofa::type::Mat<NumberOfNodesInElement, spatial_dimensions, Real_t<DataTypes>> dN_dQ { sofa::type::NOINIT };
    };

    sofa::type::vector<std::array<PrecomputedData, NumberOfQuadraturePoints>> m_precomputedData;

    void precomputeData();

    /**
     * @brief Computes the inverse of the Gram matrix (integrated N^T * N) for each element.
     * This matrix is used for the least-square projection of values from quadrature points to nodes.
     */
    void calculateElementInverseGramMatrices(const auto& elements, sofa::type::vector<ElementGramMatrix>& inverseGramMatrices);

    /**
     * @brief Projects values evaluated at quadrature points to nodal values using least-squares.
     * @param elementId Index of the element
     * @param valuesAtQuadraturePoints Array of values evaluated at each quadrature point of the element
     * @return Array of projected values at each node of the element
     */
    std::array<StressVoigtVector, NumberOfNodesInElement> projectQuadraturePointValuesToNodes(
        sofa::Size elementId,
        const std::array<StressVoigtVector, NumberOfQuadraturePoints>& valuesAtQuadraturePoints) const;

    static StressVoigtVector deviatoricStress(const StressVoigtVector& sigma);
    static Real_t<DataTypes> vonMisesStress(const StressVoigtVector& deviatoricStress);

    core::visual::DrawElementColoredMesh<ElementType> m_renderer;
};

}  // namespace sofa::component::solidmechanics::fem::elastic
