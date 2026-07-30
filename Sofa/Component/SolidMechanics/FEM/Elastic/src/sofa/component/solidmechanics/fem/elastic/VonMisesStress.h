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
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

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

    // a stress tensor represented as a vector using the Voigt mapping
    using StressVoigtVector = std::array<sofa::Real_t<DataTypes>, sofa::type::NumberOfIndependentElements<spatial_dimensions>>;

public:

    void init() override;

    // A stress value for each node in an element
    using LocalStressValues = std::array<sofa::Real_t<DataTypes>, NumberOfNodesInElement>;
    Data<sofa::type::vector<LocalStressValues>> d_nodalStress;

    Data<bool> d_continuousField;

    VonMisesStress();

    void handleEvent(core::objectmodel::Event*) override;

protected:
    using ElementMassMatrix = sofa::type::Mat<NumberOfNodesInElement, NumberOfNodesInElement, sofa::Real_t<DataTypes>>;
    sofa::type::vector<ElementMassMatrix> m_elementMassMatrices;

    void calculateElementMassMatrix(const auto& elements, sofa::type::vector<ElementMassMatrix> &elementMassMatrices);

    static StressVoigtVector deviatoricStress(const StressVoigtVector& sigma);
};

}  // namespace sofa::component::solidmechanics::fem::elastic
