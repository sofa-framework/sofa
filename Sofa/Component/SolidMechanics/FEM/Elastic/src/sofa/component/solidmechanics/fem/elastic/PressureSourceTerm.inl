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
#include <sofa/component/solidmechanics/fem/elastic/PressureSourceTerm.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
PressureSourceTerm<DataTypes, ElementType>::PressureSourceTerm()
    : l_pressure(initLink("pressure", "Nodal pressure integrated by this term."))
{
}

template <class DataTypes, class ElementType>
void PressureSourceTerm<DataTypes, ElementType>::init()
{
    BaseSourceTerm<DataTypes, ElementType>::init();

    if (this->isComponentStateInvalid())
    {
        return;
    }

    if (!l_pressure)
    {
        msg_error(this) << "The 'pressure' link must be set to a NodalPressure component. "
                           "Linked path: '" << l_pressure.getLinkedPath() << "'.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template <class DataTypes, class ElementType>
sofa::Deriv_t<DataTypes> PressureSourceTerm<DataTypes, ElementType>::evaluate(
    const QuadratureContext_t& context) const
{
    if (!l_pressure)
    {
        return Deriv{};
    }

    return elementNormal(context.jacobian) * this->interpolateProperty(*l_pressure, context);
}

template <class DataTypes, class ElementType>
sofa::type::vector<const sofa::core::objectmodel::BaseData*>
PressureSourceTerm<DataTypes, ElementType>::integrandInputs() const
{
    if (!l_pressure)
    {
        return {};
    }

    return {&l_pressure->d_property};
}

template <class DataTypes, class ElementType>
auto PressureSourceTerm<DataTypes, ElementType>::evaluateStiffness(
    const QuadratureContext_t& context, sofa::Size node) const -> SourceDerivative
{
    if (!l_pressure)
    {
        return SourceDerivative{};
    }

    const auto pressure = this->interpolateProperty(*l_pressure, context);
    const auto& gradient = context.gradientShapeFunctions[node];

    if constexpr (QuadratureContext_t::spatial_dimensions == 3)
    {
        const auto tangent0 = context.jacobian.col(0);
        const auto tangent1 = context.jacobian.col(1);

        const SourceDerivative skewTangent0 {
            {Real{0}, -tangent0[2], tangent0[1]},
            {tangent0[2], Real{0}, -tangent0[0]},
            {-tangent0[1], tangent0[0], Real{0}}};

        const SourceDerivative skewTangent1 {
            {Real{0}, -tangent1[2], tangent1[1]},
            {tangent1[2], Real{0}, -tangent1[0]},
            {-tangent1[1], tangent1[0], Real{0}}};

        return (skewTangent0 * gradient[1] - skewTangent1 * gradient[0]) * pressure;
    }
    else
    {
        const SourceDerivative rotation {{Real{0}, Real{1}}, {Real{-1}, Real{0}}};

        return rotation * (gradient[0] * pressure);
    }
}

}  // namespace sofa::component::solidmechanics::fem::elastic
