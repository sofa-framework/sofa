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
#include "SceneCheckUsingAlias.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>


namespace sofa::_scenechecking_
{

const bool SceneCheckUsingAliasRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckUsingAlias::newSPtr());

using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseObjectDescription;
using sofa::core::ObjectFactory;


SceneCheckUsingAlias::SceneCheckUsingAlias()
{
    /// Add a callback to be n
    ObjectFactory::getInstance()->setCallback([this](Base* o, BaseObjectDescription *arg) {
        const std::string typeNameInScene = arg->getAttribute("type", "");
        if ( typeNameInScene != o->getClassName() )
        {
            this->m_componentsCreatedUsingAlias[o->getClassName()].push_back(typeNameInScene);
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
        std::vector<std::string> unique_aliases(i.second);
        std::sort( unique_aliases.begin(), unique_aliases.end() );
        unique_aliases.erase( std::unique(unique_aliases.begin(), unique_aliases.end()), unique_aliases.end() );

        for(std::string &unique_alias : unique_aliases)
        {
            const unsigned int count = std::count(i.second.begin(), i.second.end(), unique_alias);

            using sofa::helper::lifecycle::ComponentChange;
            using sofa::helper::lifecycle::renamedComponents;

            usingAliasesWarning << "  - " << i.first << ": " << count << " created with alias \"" <<  unique_alias << "\"";
            auto searchAlias = renamedComponents.find(unique_alias);
            if( searchAlias != renamedComponents.end() )
            {
                usingAliasesWarning << " " << searchAlias->second.getMessage();
            }

            if(unique_alias != unique_aliases.back()) usingAliasesWarning << msgendl;
        }

        if(i.first != m_componentsCreatedUsingAlias.rbegin()->first) usingAliasesWarning << msgendl;
    }
    msg_warning(this->getName()) << usingAliasesWarning.str();

    m_componentsCreatedUsingAlias.clear();
}

} // namespace sofa::_scenechecking_
