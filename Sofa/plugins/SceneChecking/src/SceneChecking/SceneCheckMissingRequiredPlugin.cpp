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
#include "SceneCheckMissingRequiredPlugin.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/system/PluginManager.h>

#include <sofa/simulation/RequiredPlugin.h>
#include <sofa/simulation/Node.h>

namespace sofa::_scenechecking_
{

using sofa::core::objectmodel::Base;
using sofa::simulation::RequiredPlugin;
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

void SceneCheckMissingRequiredPlugin::doCheckOn(sofa::simulation::Node* node)
{
    for (const auto& object : node->object)
    {
        const ObjectFactory::ClassEntry entry = ObjectFactory::getInstance()->getEntry(object->getClassName());
        if(!entry.creatorMap.empty())
        {
            ObjectFactory::CreatorMap::const_iterator it = entry.creatorMap.find(object->getTemplateName());
            if(entry.creatorMap.end() != it && *it->second->getTarget())
            {
                const std::string pluginName = it->second->getTarget();
                const std::string path = PluginManager::getInstance().findPlugin(pluginName);
                if( PluginManager::getInstance().pluginIsLoaded(path)
                        && m_loadedPlugins.find(pluginName) == m_loadedPlugins.end() )
                {
                    m_requiredPlugins[pluginName].push_back(object->getClassName());
                }
            }
        }
    }

    //sort and remove duplicates
    for (auto& plugins : m_requiredPlugins)
    {
        auto& v = plugins.second;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
}

void SceneCheckMissingRequiredPlugin::doPrintSummary()
{
    if(!m_requiredPlugins.empty())
    {
        const std::string indent { "  "};
        std::stringstream tmp;
        for(const auto& kv : m_requiredPlugins)
        {
            tmp << indent << "<RequiredPlugin name=\""<<kv.first<<"\"/> <!-- Needed to use components [";
            if (!kv.second.empty())
            {
                std::copy(kv.second.begin(), kv.second.end() - 1, std::ostream_iterator<std::string>(tmp, ", "));
                tmp << kv.second.back();
            }
            tmp <<"] -->" << msgendl;
        }
        msg_warning(this->getName())
                << "This scene is using component defined in plugins but is not importing the required plugins." << msgendl
                << indent << "Your scene may not work on a sofa environment with different pre-loaded plugins." << msgendl
                << indent << "To fix your scene and remove this warning you just need to cut & paste the following lines at the beginning of your scene (if it is a .scn): " << msgendl
                << tmp.str();
    }
}

void SceneCheckMissingRequiredPlugin::doInit(sofa::simulation::Node* node)
{
    sofa::type::vector< RequiredPlugin* > plugins;
    node->getTreeObjects< RequiredPlugin >(&plugins);

    m_requiredPlugins.clear();
    m_loadedPlugins.clear();

    for(auto& plugin : plugins)
    {
        for(auto& pluginName : plugin->d_loadedPlugins.getValue())
        {
            m_loadedPlugins[pluginName] = true;
        }
    }
}

} // namespace sofa::_scenechecking_
