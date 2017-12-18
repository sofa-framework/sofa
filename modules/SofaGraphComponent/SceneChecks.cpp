/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/version.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>

#include "SceneChecks.h"
#include "RequiredPlugin.h"

#include "APIVersion.h"
using sofa::component::APIVersion ;

namespace sofa
{
namespace simulation
{
namespace _scenechecks_
{

using sofa::core::objectmodel::Base ;
using sofa::component::misc::RequiredPlugin ;
using sofa::core::ObjectFactory ;
using sofa::core::ExecParams ;
using sofa::helper::system::PluginRepository ;
using sofa::helper::system::PluginManager ;

const std::string SceneCheckDuplicatedName::getName()
{
    return "SceneCheckDuplicatedName";
}

const std::string SceneCheckDuplicatedName::getDesc()
{
    return "Check there is not duplicated name in the scenegraph";
}

void SceneCheckDuplicatedName::doCheckOn(Node* node)
{
    std::map<std::string, int> duplicated ;
    for (auto& object : node->object )
    {
        if( duplicated.find(object->getName()) == duplicated.end() )
            duplicated[object->getName()] = 0 ;
        duplicated[object->getName()]++ ;
    }

    for (auto& child : node->child )
    {
        if( duplicated.find(child->getName()) == duplicated.end() )
            duplicated[child->getName()] = 0 ;
        duplicated[child->getName()]++ ;
    }

    std::stringstream tmp ;
    for(auto& p : duplicated)
    {
        if(p.second!=1)
        {
            tmp << "- duplicated '" << p.first << "'" << msgendl ;
        }
    }
    if(!tmp.str().empty())
    {
       msg_warning("SceneCheckDuplicatedName") << "In '"<<  node->getPathName() <<"'" << msgendl
                                               << tmp.str() ;
    }

}

const std::string SceneCheckMissingRequiredPlugin::getName()
{
    return "SceneCheckMissingRequiredPlugin";
}

const std::string SceneCheckMissingRequiredPlugin::getDesc()
{
    return "Check for each component provided by a plugin that the corresponding <RequiredPlugin> directive is present in the scene";
}

void SceneCheckMissingRequiredPlugin::doCheckOn(Node* node)
{
    for (auto& object : node->object )
    {
        ObjectFactory::ClassEntry entry = ObjectFactory::getInstance()->getEntry(object->getClassName());
        if(!entry.creatorMap.empty())
        {
            ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(object->getTemplateName());
            if(entry.creatorMap.end() != it && *it->second->getTarget()){
                std::string pluginName = it->second->getTarget() ;
                std::string path = PluginManager::getInstance().findPlugin(pluginName) ;
                if( PluginManager::getInstance().pluginIsLoaded(path)
                        && m_requiredPlugins.find(pluginName) == m_requiredPlugins.end() )
                {
                    msg_warning("SceneChecker")
                            << "This scene is using component '" << object->getClassName() << "'. " << msgendl
                            << "This component is part of the '" << pluginName << "' plugin but there is no <RequiredPlugin name='" << pluginName << "' /> directive in your scene." << msgendl
                            << "Your scene may not work on a sofa environment that does not have pre-loaded the plugin." << msgendl
                            << "To fix your scene and remove this warning you need to add the RequiredPlugin directive at the beginning of your scene. ";
                }
            }
        }
    }
}

void SceneCheckMissingRequiredPlugin::doInit(Node* node)
{
    helper::vector< RequiredPlugin* > plugins ;
    node->getTreeObjects< RequiredPlugin >(&plugins) ;

    for(auto& plugin : plugins)
        m_requiredPlugins[plugin->getName()] = true ;
}


} // _scenechecks_

} // namespace simulation

} // namespace sofa

