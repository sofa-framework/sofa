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
#include <sofa/component/solidmechanics/fem/elastic/VectorSourceTerm.h>
#include <sofa/component/solidmechanics/fem/elastic/NodalPropertyInterpolation.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
VectorSourceTerm<DataTypes, ElementType>::VectorSourceTerm()
    : l_sourceDensity(initLink("sourceDensity", "Nodal source density integrated by this term."))
{
}

template <class DataTypes, class ElementType>
void VectorSourceTerm<DataTypes, ElementType>::init()
{
    BaseSourceTerm<DataTypes, ElementType>::init();

    if (this->isComponentStateInvalid())
    {
        return;
    }

    if (!l_sourceDensity)
    {
        msg_error(this) << "The 'sourceDensity' link must be set to a NodalSourceDensity "
                           "component. Linked path: '" << l_sourceDensity.getLinkedPath() << "'.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template <class DataTypes, class ElementType>
sofa::Deriv_t<DataTypes> VectorSourceTerm<DataTypes, ElementType>::evaluate(
    const QuadratureContext& context) const
{
    if (!l_sourceDensity)
    {
        return Deriv{};
    }

    return interpolateNodalProperty(*l_sourceDensity, context);
}

}  // namespace sofa::component::solidmechanics::fem::elastic
