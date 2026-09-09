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

#include <sofa/core/BaseNodalProperty.h>
#include <sofa/helper/accessor.h>
#include <sofa/type/vector.h>

#include <array>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @brief Value of a nodal property interpolated at a quadrature point.
 *
 * The property is gathered at the nodes of the element being integrated and combined with the
 * shape functions evaluated at that point.
 *
 * @param property The component holding the nodal values.
 * @param context Geometry of the quadrature point.
 */
template <class PropertyType, class QuadratureContext>
PropertyType interpolateNodalProperty(
    const sofa::core::BaseNodalProperty<PropertyType>& property,
    const QuadratureContext& context)
{
    using FiniteElement = typename QuadratureContext::FiniteElement;

    static constexpr sofa::Size NumberOfNodesInElement = QuadratureContext::NumberOfNodesInElement;

    sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<PropertyType>>> propertyAccessor {
        property.d_property};

    std::array<PropertyType, NumberOfNodesInElement> elementNodesProperty;
    for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
    {
        elementNodesProperty[i] = property.getNodeProperty(context.element[i], propertyAccessor);
    }

    return FiniteElement::Helper::evaluateValueInElement(elementNodesProperty, context.N);
}

}  // namespace sofa::component::solidmechanics::fem::elastic
