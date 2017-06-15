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

#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>

using sofa::helper::system::PluginManager;

static std::string pluginName = "TestPlugin" ;

#ifdef NDEBUG
static std::string pluginFileName = "TestPlugin" ;
#else
static std::string pluginFileName = "TestPlugin_d" ;
#endif //N_DEBUG

static std::string nonpluginName = "RandomNameForAPluginButHopeItDoesNotExist";

const std::string dotExt = "." + sofa::helper::system::DynamicLibrary::extension;
#ifdef WIN32
const std::string separator = "\\";
const std::string prefix = "";
#else
const std::string separator = "/";
const std::string prefix = "lib";
#endif // WIN32

struct PluginManager_test: public ::testing::Test
{
    std::string pluginDir;

    void SetUp()
    {
        // Add the plugin directory to PluginRepository
#ifdef WIN32
        pluginDir = sofa::helper::Utils::getExecutableDirectory();
#else
        pluginDir = sofa::helper::Utils::getSofaPathPrefix() + "/lib";
#endif
        sofa::helper::system::PluginRepository.addFirstPath(pluginDir);
    }
};


TEST_F(PluginManager_test, loadTestPluginByPath)
{
    sofa::helper::system::PluginManager&pm = sofa::helper::system::PluginManager::getInstance();

    std::string pluginPath = pluginDir + separator + prefix + pluginFileName + dotExt;
    std::string nonpluginPath = pluginDir + separator + prefix + nonpluginName + dotExt;

    std::cout << "Loading plugin: " << pluginPath << std::endl;

    ASSERT_TRUE(pm.loadPluginByPath(pluginPath));
    ASSERT_FALSE(pm.loadPluginByPath(nonpluginPath));

    ASSERT_GT(pm.findPlugin(pluginName).size(), 0u);
    ASSERT_EQ(pm.findPlugin(nonpluginName).size(), 0u);

    //It is better to unload the testPlugin in each test or it will stay loaded for the entire fixture
    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);

}

TEST_F(PluginManager_test, loadTestPluginByName )
{
    sofa::helper::system::PluginManager&pm = sofa::helper::system::PluginManager::getInstance();

    ASSERT_TRUE(pm.loadPluginByName(pluginName) );
    ASSERT_FALSE(pm.loadPluginByName(nonpluginName));

    std::string pluginPath = pm.findPlugin(pluginName);
    ASSERT_GT(pluginPath.size(), 0u);
    ASSERT_EQ(pm.findPlugin(nonpluginName).size(), 0u);

    //Same
    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);
}

TEST_F(PluginManager_test, pluginEntries)
{
    sofa::helper::system::PluginManager&pm = sofa::helper::system::PluginManager::getInstance();

    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);
    sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    EXPECT_TRUE(p.initExternalModule.func != NULL);
    EXPECT_TRUE(p.getModuleName.func != NULL);
    EXPECT_TRUE(p.getModuleVersion.func != NULL);
    EXPECT_TRUE(p.getModuleLicense.func != NULL);
    EXPECT_TRUE(p.getModuleDescription.func != NULL);
    EXPECT_TRUE(p.getModuleComponentList.func != NULL);

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);
}

TEST_F(PluginManager_test, pluginEntriesValues)
{
    sofa::helper::system::PluginManager&pm = sofa::helper::system::PluginManager::getInstance();

    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);
    sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    std::string testModuleName = "TestPlugin";
    std::string testModuleVersion = "0.7";
    std::string testModuleLicence = "LicenceTest";
    std::string testModuleDescription = "Description of the Test Plugin";
    std::string testModuleComponentList = "ComponentA, ComponentB";

    ASSERT_EQ(0, std::string(p.getModuleName()).compare(testModuleName));
    ASSERT_NE(0, std::string(p.getModuleName()).compare(testModuleName + "azerty"));

    ASSERT_EQ(0, std::string(p.getModuleVersion()).compare(testModuleVersion));
    ASSERT_NE(0, std::string(p.getModuleVersion()).compare(testModuleVersion + "77777"));

    ASSERT_EQ(0, std::string(p.getModuleLicense()).compare(testModuleLicence));
    ASSERT_NE(0, std::string(p.getModuleLicense()).compare(testModuleLicence + "GPLBSDProprio"));

    ASSERT_EQ(0, std::string(p.getModuleDescription()).compare(testModuleDescription));
    ASSERT_NE(0, std::string(p.getModuleDescription()).compare(testModuleDescription + "blablablabalbal"));

    ASSERT_EQ(0, std::string(p.getModuleComponentList()).compare(testModuleComponentList));
    ASSERT_NE(0, std::string(p.getModuleComponentList()).compare(testModuleComponentList + "ComponentZ"));

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);
}

TEST_F(PluginManager_test, testIniFile)
{
    sofa::helper::system::PluginManager&pm = sofa::helper::system::PluginManager::getInstance();
    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);

    const std::string pathIniFile = std::string(FRAMEWORK_TEST_RESOURCES_DIR) + separator + "PluginManager_test.ini";
    pm.writeToIniFile(pathIniFile);

    //writeToIniFile does not return anything to say if the file was created without error...
    ASSERT_TRUE(sofa::helper::system::FileSystem::exists(pathIniFile));

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    ASSERT_TRUE(sofa::helper::system::FileSystem::exists(pathIniFile));

    pm.readFromIniFile(pathIniFile);
    ASSERT_EQ(pm.findPlugin(pluginName).compare(pluginPath), 0);

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);
}
