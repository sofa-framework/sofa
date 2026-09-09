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
#include <sofa/component/solidmechanics/fem/elastic/GeometricSourceTerm.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType, class PropertyType>
GeometricSourceTerm<DataTypes, ElementType, PropertyType>::GeometricSourceTerm()
    : l_nodalProperty(initLink("nodalProperty", "Nodal values of the property this source term is "
                "built from."))
{
}

template <class DataTypes, class ElementType, class PropertyType>
void GeometricSourceTerm<DataTypes, ElementType, PropertyType>::init()
{
    BaseGeometricSourceTerm<DataTypes, ElementType>::init();

    if (this->isComponentStateInvalid())
    {
        return;
    }

    if (!l_nodalProperty)
    {
        msg_error(this) << "The 'nodalProperty' link must be set to a BaseNodalProperty component. "
                           "Linked path: '" << l_nodalProperty.getLinkedPath() << "'.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template <class DataTypes, class ElementType, class PropertyType>
PropertyType GeometricSourceTerm<DataTypes, ElementType, PropertyType>::interpolateProperty(
    const QuadratureContext& context) const
{
    sofa::helper::ReadAccessor propertyAccessor { l_nodalProperty->d_property };

    std::array<PropertyType, NumberOfNodesInElement> elementNodesProperty;
    for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
    {
        elementNodesProperty[i] =
            l_nodalProperty->getNodeProperty(context.element[i], propertyAccessor);
    }

    return FiniteElement::Helper::evaluateValueInElement(elementNodesProperty, context.N);
}

}  // namespace sofa::component::solidmechanics::fem::elastic
