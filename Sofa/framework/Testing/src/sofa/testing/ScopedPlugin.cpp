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
#include <sofa/core/ObjectFactory.h>
#include <sofa/testing/ScopedPlugin.h>

namespace sofa::testing
{

ScopedPlugin::ScopedPlugin(const std::string& pluginName, const bool unloadAllPlugins,
                           helper::system::PluginManager* pluginManager)
: m_pluginManager(pluginManager), m_pluginName(pluginName), m_unloadAllPlugins(unloadAllPlugins)
{
    if (m_pluginManager)
    {
        m_status = pluginManager->loadPlugin(pluginName);
        if(m_status == helper::system::PluginManager::PluginLoadStatus::SUCCESS)
        {
            sofa::core::ObjectFactory::getInstance()->registerObjectsFromPlugin(pluginName);
        }
    }
}

ScopedPlugin::~ScopedPlugin()
{
    if (m_pluginManager)
    {
        if (m_unloadAllPlugins)
        {
            for(const auto& [loadedPath, loadedPlugin] : m_pluginManager->getPluginMap())
            {
                m_pluginManager->unloadPlugin(loadedPath);
            }
        }
        else
        {
            if (m_status == helper::system::PluginManager::PluginLoadStatus::SUCCESS)
            {
                const auto [path, isLoaded] = m_pluginManager->isPluginLoaded(m_pluginName);
                if (isLoaded)
                {
                    m_pluginManager->unloadPlugin(path);
                }
            }
        }
    }
}

helper::system::PluginManager::PluginLoadStatus ScopedPlugin::getStatus() const
{
    return m_status;
}
}
