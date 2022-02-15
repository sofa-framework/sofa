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

#include <SofaBaseUtils/RequiredPlugin.h>
using sofa::component::misc::RequiredPlugin;

#include <sofa/component/sceneutility/ImportComponent.h>
using sofa::component::sceneutility::ImportComponent;

#include <sofa/simulation/Node.h>

namespace sofa::simulation::_scenechecking_
{

using sofa::core::objectmodel::Base;
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
    for (const auto& object : node->object)
    {
        const ObjectFactory::ClassEntry& entry = ObjectFactory::getInstance()->getEntry(object->getClass());
        const std::string pluginName = entry.compilationTarget;
        const std::string path = PluginManager::getInstance().findPlugin(pluginName);
        if( PluginManager::getInstance().pluginIsLoaded(path)
                && m_loadedPlugins.find(pluginName) == m_loadedPlugins.end() )
        {
            m_requiredPlugins[pluginName].push_back(object->getClassName());
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

void SceneCheckMissingRequiredPlugin::doInit(Node* node)
{
    // Fill the m_loadedPlugin map that is then used in the other function
    // there is two source of plugin, one is the component <RequiredPlugin> the other is
    // <Import>.
    // For that we first scan the scene graph to search using getTreeObject.
    // As this is a templated function we need to specify it by the type of
    // component we are searching for.
    //
    // Once this is done we add into the m_loadedPlugins map the loaded plugin's name.
    type::vector< RequiredPlugin* > plugins;
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

    type::vector< ImportComponent* > imports;
    node->getTreeObjects< ImportComponent >(&imports);
    for(auto& plugin : imports)
    {
        m_loadedPlugins[plugin->d_fromPlugin.getValue()] = true;
    }

}

} // namespace sofa::simulation::_scenechecking_
