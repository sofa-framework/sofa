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
#include <sofa/testing/ScopedPlugin.h>
#include <gtest/gtest.h>

TEST(ScopedPlugin, test)
{
    static std::string pluginName = "Sofa.Component.AnimationLoop";
    auto& pluginManager = sofa::helper::system::PluginManager::getInstance();

    //make sure that pluginName is not already loaded
    {
        const auto [path, isLoaded] = pluginManager.isPluginLoaded(pluginName);
        if (isLoaded)
        {
            pluginManager.unloadPlugin(path);
        }
    }

    {
        const auto [path, isLoaded] = pluginManager.isPluginLoaded(pluginName);
        EXPECT_FALSE(isLoaded);
    }

    {
        const sofa::testing::ScopedPlugin plugin(pluginName);

        {
            const auto [path, isLoaded] = pluginManager.isPluginLoaded(pluginName);
            EXPECT_TRUE(isLoaded);
        }

        //end of scope: plugin should be unloaded
    }

    {
        const auto [path, isLoaded] = pluginManager.isPluginLoaded(pluginName);
        EXPECT_FALSE(isLoaded);
    }
}
