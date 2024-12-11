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
#include <sofa/testing/config.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::testing
{

struct SOFA_TESTING_API ScopedPlugin
{
    ScopedPlugin() = delete;
    explicit ScopedPlugin(
        const std::string& pluginName, bool unloadAllPlugins = true,
        helper::system::PluginManager* pluginManager = &helper::system::PluginManager::getInstance());
    ~ScopedPlugin();

    helper::system::PluginManager::PluginLoadStatus getStatus() const;

private:
    helper::system::PluginManager* m_pluginManager { nullptr };
    std::string m_pluginName;
    helper::system::PluginManager::PluginLoadStatus m_status;
    bool m_unloadAllPlugins { true };
};


}
