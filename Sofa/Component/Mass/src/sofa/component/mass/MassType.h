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

#include <sofa/component/mass/config.h>
#include <sofa/core/behavior/BaseMass.h>

namespace sofa::component::mass
{
/*
 * This (empty) templated struct is used for determining a type of mass according to
 * the associated DataType. 
 * The generic version of it does not contain any type/definition,  
 * and will provoke an error if one is trying to determine a MassType without having
 * specialized this struct first.
 * For example, MassType specialized on Vec<N,Real> should return Real as its type.
 * (see VecMassType.h)
 * 
 * This is used by the Mass components to find a MassType according to their DataType.
 */
template<typename DataType>
struct MassType
{
    // if you want to associate a mass type YourType for a particular DataType
    // using type = YourType;
};


/**
 * Function used in parsing some classes derived from BaseMass to warn the user how the template
 * attributes have changed since #2644
 */
template<class MassType>
void parseMassTemplate(sofa::core::objectmodel::BaseObjectDescription* arg, const core::behavior::BaseMass* mass)
{
    if (arg->getAttribute("template"))
    {
        const auto splitTemplates = sofa::helper::split(std::string(arg->getAttribute("template")), ',');
        if (splitTemplates.size() > 1)
        {
            // check if the given 2nd template is the deprecated MassType one
            if (splitTemplates[1] == "float" || splitTemplates[1] == "double" || splitTemplates[1].find("RigidMass") != std::string::npos)
            {
                msg_warning(mass) << "MassType is not required anymore and the template is deprecated, please delete it from your scene." << msgendl
                    << "As your mass is templated on '" << mass->getTemplateName() << "', MassType has been defined as " << sofa::helper::NameDecoder::getTypeName<MassType>() << " .";
                msg_warning(mass) << "If you want to set the template, you must write now \"template='" << mass->getTemplateName() << "'\" .";
            }
        }
    }
}
} // namespace sofa::component::mass
