/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SceneCheckUsingAlias.h"

#include <sofa/version.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>


namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{

using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseObjectDescription;
using sofa::core::ObjectFactory;


SceneCheckUsingAlias::SceneCheckUsingAlias()
{
    /// Add a callback to be n
    ObjectFactory::getInstance()->setCallback([this](Base* o, BaseObjectDescription *arg) {
        if (o->getClassName() != arg->getAttribute("type", "") )
        {
            std::string alias = arg->getAttribute("type", "");

            std::vector<std::string> v = this->m_componentsCreatedUsingAlias[o->getClassName()];
            if ( std::find(v.begin(), v.end(), alias) == v.end() )
            {
                this->m_componentsCreatedUsingAlias[o->getClassName()].push_back(alias);
            }
        }
    });
}

SceneCheckUsingAlias::~SceneCheckUsingAlias()
{

}

const std::string SceneCheckUsingAlias::getName()
{
    return "SceneCheckUsingAlias";
}

const std::string SceneCheckUsingAlias::getDesc()
{
    return "Check if a Component has been created using an Alias.";
}

void SceneCheckUsingAlias::doPrintSummary()
{
    if ( this->m_componentsCreatedUsingAlias.empty() )
    {
        return;
    }
    
    std::stringstream usingAliasesWarning;
    usingAliasesWarning << "This scene is using hard coded aliases. Aliases can be very confusing, "
                           "use with caution." << msgendl;
    for (auto i : this->m_componentsCreatedUsingAlias)
    {
        if (i.second.size() > 1)
            usingAliasesWarning << "  - " << i.first << " have been created using the aliases ";
        else
            usingAliasesWarning << "  - " << i.first << " has been created using the alias ";

        bool first = true;
        for (std::string &alias : i.second)
        {
            if (first)
                usingAliasesWarning << "\"" << alias << "\"";
            else
                usingAliasesWarning << ", \"" << alias << "\"";

            first = false;
        }
        usingAliasesWarning << "." << msgendl;
    }
    msg_warning(this->getName()) << usingAliasesWarning.str();
}

} // namespace _scenechecking_
} // namespace simulation
} // namespace sofa
