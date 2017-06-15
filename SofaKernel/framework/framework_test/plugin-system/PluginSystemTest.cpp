/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <sofa/core/PluginManager.h>
#include <sofa/core/Plugin.h>
#include <gtest/gtest.h>

#include <sofa/helper/system/FileSystem.h>
#include "PluginMonitor/PluginMonitor.h"

using namespace sofa::helper::system;
using sofa::core::PluginManager;

const std::string testPluginsDir(std::string(SOFA_TEST_PLUGINS_DIR)+"/");

TEST(PluginManagerTest, refreshPluginInfo)
{
    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(1, pluginManager.getPluginDirectories().size());
    EXPECT_EQ(testPluginsDir, pluginManager.getPluginDirectories()[0]);
    EXPECT_EQ(13, pluginManager.getComponentDatabase().size());
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
    EXPECT_EQ(1, PluginA_loaded);
    EXPECT_EQ(0, PluginB_loaded);
    // PluginE is linked against PluginC, it should be opened twice.
    EXPECT_EQ(2, PluginC_loaded);
    EXPECT_EQ(1, PluginD_loaded);
    EXPECT_EQ(1, PluginE_loaded);

    EXPECT_EQ(1, PluginA_unloaded);
    EXPECT_EQ(0, PluginB_unloaded);
    // PluginE is linked against PluginC, it should be opened twice.
    EXPECT_EQ(2, PluginC_unloaded);
    EXPECT_EQ(1, PluginD_unloaded);
    EXPECT_EQ(1, PluginE_unloaded);

    // Content of PluginA
    EXPECT_TRUE(pluginManager.canFindComponent("Foo"));
    EXPECT_TRUE(pluginManager.canFindComponent("Bar"));
    EXPECT_TRUE(pluginManager.canFindComponent("Bar", "float"));
    EXPECT_TRUE(pluginManager.canFindComponent("Bar", "double"));

    // Content of PluginD
    EXPECT_TRUE(pluginManager.canFindComponent("FooD"));
    EXPECT_TRUE(pluginManager.canFindComponent("Bar", "VecD"));
    EXPECT_TRUE(pluginManager.canFindComponent("BazD"));
    EXPECT_TRUE(pluginManager.canFindComponent("BazD", "float"));
    EXPECT_TRUE(pluginManager.canFindComponent("BazD", "double"));
}

TEST(PluginManagerTest, load_unload_PluginA)
{
    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(1, PluginA_loaded);
    EXPECT_EQ(1, PluginA_unloaded);
    pluginManager.loadPlugin("PluginA");
    EXPECT_EQ(2, PluginA_loaded);
    EXPECT_EQ(1, PluginA_unloaded);
    EXPECT_EQ(1, pluginManager.getLoadedPlugins().size());
    pluginManager.unloadPlugin("PluginA");
    EXPECT_EQ(2, PluginA_loaded);
    EXPECT_EQ(2, PluginA_unloaded);
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
}

TEST(PluginManagerTest, load_legacy_PluginB)
{
    // God this is ugly.
#ifdef WIN32
    std::string dir(SOFA_BIN_DIR);
#else
    std::string dir(SOFA_LIB_DIR);
#endif
    std::string possibleNames[] = {"libPluginB.so", "libPluginB.dylib", "PluginB.dll"};
    std::string PluginB_file;
    for (size_t i=0 ; i<2 ; i++) {
        std::string file(dir + "/" + possibleNames[i]);
        if (FileSystem::exists(file)) {
            PluginB_file = file;
            break;
        }
    }
	ASSERT_FALSE(PluginB_file.empty());
    //if (PluginB_file.empty()) {
    //    std::cout << "PluginB not found" << std::endl;
    //    return;
    //}

    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    EXPECT_EQ(0, PluginB_loaded);
    EXPECT_EQ(0, PluginB_unloaded);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(0, PluginB_loaded);
    EXPECT_EQ(0, PluginB_unloaded);
    pluginManager.loadPlugin(PluginB_file);
    EXPECT_EQ(1, PluginB_loaded);
    EXPECT_EQ(0, PluginB_unloaded);
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
    EXPECT_EQ(1, pluginManager.getLoadedLegacyPlugins().size());
    pluginManager.unloadAllPlugins();
    EXPECT_EQ(1, PluginB_loaded);
    EXPECT_EQ(0, PluginB_unloaded);
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
    EXPECT_EQ(1, pluginManager.getLoadedLegacyPlugins().size());
}

TEST(PluginManagerTest, load_unload_PluginD_addTemplateInstance)
{
    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(1, PluginA_loaded);
    EXPECT_EQ(1, PluginA_unloaded);
    EXPECT_EQ(1, PluginD_loaded);
    EXPECT_EQ(1, PluginD_unloaded);
    pluginManager.loadPlugin("PluginD");
    EXPECT_EQ(1, PluginA_loaded);
    EXPECT_EQ(1, PluginA_unloaded);
    EXPECT_EQ(2, PluginD_loaded);
    EXPECT_EQ(1, PluginD_unloaded);
    EXPECT_EQ(1, pluginManager.getLoadedPlugins().size());
    pluginManager.unloadPlugin("PluginD");
    EXPECT_EQ(1, PluginA_loaded);
    EXPECT_EQ(1, PluginA_unloaded);
    EXPECT_EQ(2, PluginD_loaded);
    EXPECT_EQ(2, PluginD_unloaded);
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
}

TEST(PluginManagerTest, load_unload_PluginE_linked_against_PluginC)
{
    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    EXPECT_EQ(0, PluginC_loaded);
    EXPECT_EQ(0, PluginC_unloaded);
    EXPECT_EQ(0, PluginE_loaded);
    EXPECT_EQ(0, PluginE_unloaded);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(2, PluginC_loaded);
    EXPECT_EQ(2, PluginC_unloaded);
    EXPECT_EQ(1, PluginE_loaded);
    EXPECT_EQ(1, PluginE_unloaded);
    pluginManager.loadPlugin("PluginE");
    EXPECT_EQ(3, PluginC_loaded);
    EXPECT_EQ(2, PluginC_unloaded);
    EXPECT_EQ(2, PluginE_loaded);
    EXPECT_EQ(1, PluginE_unloaded);
    EXPECT_EQ(1, pluginManager.getLoadedPlugins().size());
    pluginManager.unloadPlugin("PluginE");
    EXPECT_EQ(3, PluginC_loaded);
    EXPECT_EQ(3, PluginC_unloaded);
    EXPECT_EQ(2, PluginE_loaded);
    EXPECT_EQ(2, PluginE_unloaded);
    EXPECT_EQ(0, pluginManager.getLoadedPlugins().size());
}

TEST(PluginManagerTest, load_unload_PluginF_that_cant_be_unloaded)
{
    reset_plugin_monitor();
    PluginManager pluginManager(testPluginsDir);
    EXPECT_EQ(0, PluginF_loaded);
    EXPECT_EQ(0, PluginF_unloaded);
    pluginManager.refreshPluginInfo();
    EXPECT_EQ(1, PluginF_loaded);
    EXPECT_EQ(1, PluginF_unloaded);
    pluginManager.loadPlugin("PluginF");
    EXPECT_EQ(2, PluginF_loaded);
    EXPECT_EQ(1, PluginF_unloaded);
    EXPECT_EQ(1, pluginManager.getLoadedPlugins().size());
    EXPECT_FALSE(pluginManager.unloadPlugin("PluginF"));
    EXPECT_EQ(2, PluginF_loaded);
    EXPECT_EQ(1, PluginF_unloaded);
    pluginManager.unloadAllPlugins();
    EXPECT_EQ(2, PluginF_loaded);
    EXPECT_EQ(1, PluginF_unloaded);
    EXPECT_EQ(1, pluginManager.getLoadedPlugins().size());
}
