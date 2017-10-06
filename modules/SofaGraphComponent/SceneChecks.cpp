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
                            << "This component is part of the '" << pluginName << "' plugin but there is no <RequiredPlugin name='" << pluginName << "'> directive in your scene." << msgendl
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


const std::string SceneCheckAPIChange::getName()
{
    return "SceneCheckAPIChange";
}

const std::string SceneCheckAPIChange::getDesc()
{
    return "Check for each component that the behavior have not changed since reference version of sofa.";
}

void SceneCheckAPIChange::doInit(Node* node)
{
    std::stringstream version;
    version << SOFA_VERSION / 10000 << "." << SOFA_VERSION / 100 % 100;
    m_currentApiLevel = version.str();

    APIVersion* apiversion {nullptr} ;
    /// 1. Find if there is an APIVersion component in the scene. If there is none, warn the user and set
    /// the version to 17.06 (the last version before it was introduced). If there is one...use
    /// this component to request the API version requested by the scene.
    node->getTreeObject(apiversion) ;
    if(!apiversion)
    {
        msg_info("SceneChecker") << "The 'APIVersion' directive is missing in the current scene. Switching to the default APIVersion level '"<< m_selectedApiLevel <<"' " ;
    }
    else
    {
        m_selectedApiLevel = apiversion->getApiLevel() ;
    }
}

void SceneCheckAPIChange::doCheckOn(Node* node)
{
    for (auto& object : node->object )
    {
        if(m_selectedApiLevel != m_currentApiLevel && m_changesets.find(m_selectedApiLevel) != m_changesets.end())
        {
            for(auto& hook : m_changesets[m_selectedApiLevel])
            {
                hook(object.get());
            }
        }

    }
}

void SceneCheckAPIChange::installDefaultChangeSets()
{
    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "RestShapeSpringsForceField" && o->findData("external_rest_shape")->isSet())
            msg_warning(o) << "RestShapeSpringsForceField have changed since 17.06. The parameter 'external_rest_shape' is now a Link. To fix your scene you need to add and '@' in front of the provided path. See PR#315" ;
    }) ;

    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "BoxStiffSpringForceField" )
            msg_warning(o) << "BoxStiffSpringForceField have changed since 17.06. To use the old behavior you need to set parameter 'forceOldBehavior=true'" ;
    }) ;

    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "TheComponentWeWantToRemove" )
            msg_warning(o) << "TheComponentWewantToRemove is deprecated since sofa 17.06. It have been replaced by TheSuperComponent. #See PR318" ;
    }) ;
}

void SceneCheckAPIChange::addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct)
{
    m_changesets[version].push_back(fct) ;
}


} // _scenechecks_

} // namespace simulation

} // namespace sofa

