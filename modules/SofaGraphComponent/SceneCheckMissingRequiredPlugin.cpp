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
#include "SceneCheckMissingRequiredPlugin.h"

#include <sofa/version.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/system/PluginManager.h>

#include "RequiredPlugin.h"

namespace sofa
{
namespace simulation
{
namespace _scenechecks_
{

using sofa::core::objectmodel::Base;
using sofa::component::misc::RequiredPlugin;
using sofa::core::ObjectFactory;
using sofa::helper::system::PluginManager;

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
                std::string pluginName = it->second->getTarget();
                std::string path = PluginManager::getInstance().findPlugin(pluginName);
                if( PluginManager::getInstance().pluginIsLoaded(path)
                        && m_loadedPlugins.find(pluginName) == m_loadedPlugins.end() )
                {
                    if( m_requiredPlugins.empty() ){
                        m_requiredPlugins[pluginName].push_back(object->getClassName());
                    }else{
                        std::vector<std::string>& t = m_requiredPlugins[pluginName];
                        if( std::find(t.begin(), t.end(), object->getClassName()) ==t.end() )
                            t.push_back(object->getClassName());
                    }
                }
            }
        }
    }
}

void SceneCheckMissingRequiredPlugin::doPrintSummary()
{
    if(!m_requiredPlugins.empty())
    {
        std::stringstream tmp;
        for(auto& kv : m_requiredPlugins)
        {
            tmp << "<RequiredPlugin pluginName='"<<kv.first<<"'/> <!-- Needed to use components [";
            for(auto& name : kv.second)
            {
                tmp << name << ", ";
            }
            tmp <<"]-->";
        }
        msg_warning(this->getName())
                << "This scene is using component defined in plugins but is not importing the required plugins." << msgendl
                << "  " << "Your scene may not work on a sofa environment with different pre-loaded plugins." << msgendl
                << "  " << "To fix your scene and remove this warning you just need to cut & paste the following lines at the begining of your scene (if it is a .scn): " << msgendl
                << "  " << tmp.str();
    }
}

void SceneCheckMissingRequiredPlugin::doInit(Node* node)
{
    helper::vector< RequiredPlugin* > plugins;
    node->getTreeObjects< RequiredPlugin >(&plugins);

    m_requiredPlugins.clear();
    m_loadedPlugins.clear();

    for(auto& plugin : plugins)
    {
        for(auto& pluginName : plugin->d_pluginName.getValue())
        {
            m_loadedPlugins[pluginName] = true;
        }
    }
}


} // _scenechecks_

} // namespace simulation

} // namespace sofa

