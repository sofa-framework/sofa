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
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <fstream>

using sofa::helper::system::PluginManager;
using sofa::helper::system::FileSystem;

static std::string pluginName = "TestPlugin";

#ifdef NDEBUG
static std::string pluginFileName = "TestPlugin";
#else
static std::string pluginFileName = "TestPlugin_d";
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

struct PluginManager_test: public BaseTest
{
    std::string pluginDir;

    // This list of paths will be deleted when cleaning-up the test
    sofa::type::vector<std::string> createdFilesToDelete;

    void SetUp() override
    {
        // Set pluginDir by searching pluginFileName in the PluginRepository
        for ( std::string path : sofa::helper::system::PluginRepository.getPaths() )
        {
            if ( FileSystem::exists(path + separator + prefix + pluginFileName + dotExt) )
            {
                pluginDir = path;
                break;
            }
        }

        ASSERT_FALSE( pluginDir.empty() );

        std::cout << "PluginManager_test.loadTestPluginByPath: "
                  << "pluginDir = " << pluginDir
                  << std::endl;
    }

    void TearDown() override
    {
        for (const auto& file : createdFilesToDelete)
        {
            EXPECT_TRUE(FileSystem::removeFile(file));
        }

        PluginManager&pm = PluginManager::getInstance();
        //empty loaded plugin(s)
        std::vector<std::string> toDelete;
        for (const auto& it : pm.getPluginMap())
        {
            toDelete.push_back(it.first);
        }

        for(const std::string& p : toDelete)
        {
            ASSERT_TRUE(pm.unloadPlugin(p));
        }

        ASSERT_EQ(pm.getPluginMap().size(), 0u);
    }
};


TEST_F(PluginManager_test, loadTestPluginByPath)
{
    PluginManager&pm = PluginManager::getInstance();

    const std::string pluginPath = pluginDir + separator + prefix + pluginFileName + dotExt;
    const std::string nonpluginPath = pluginDir + separator + prefix + nonpluginName + dotExt;

    std::cout << "PluginManager_test.loadTestPluginByPath: "
              << "pluginPath = " << pluginPath
              << std::endl;

    /// Check that existing plugins are correctly handled and returns no
    /// error/warning message.
    {
        EXPECT_MSG_NOEMIT(Warning, Error);

        std::cout << "PluginManager_test.loadTestPluginByPath: "
                  << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
                  << std::endl;
        ASSERT_EQ(pm.loadPluginByPath(pluginPath), PluginManager::PluginLoadStatus::SUCCESS);
        ASSERT_GT(pm.findPlugin(pluginName).size(), 0u);
    }

    /// Check that non existing plugin are currectly handled and returns an
    /// error message.
    {
        EXPECT_MSG_NOEMIT(Warning);
        EXPECT_MSG_EMIT(Error);

        std::cout << "PluginManager_test.loadTestPluginByPath: "
                  << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
                  << std::endl;
        ASSERT_EQ(pm.loadPluginByPath(nonpluginPath), PluginManager::PluginLoadStatus::PLUGIN_FILE_NOT_FOUND);
        ASSERT_EQ(pm.findPlugin(nonpluginName).size(), 0u);
        std::cout << "PluginManager_test.loadTestPluginByPath: "
                  << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
                  << std::endl;
    }
}

TEST_F(PluginManager_test, loadTestPluginByName )
{
    PluginManager&pm = PluginManager::getInstance();

    /// Check that existing plugins are correctly handled and returns no
    /// error/warning message.
    {
        EXPECT_MSG_NOEMIT(Warning, Error);

        ASSERT_EQ(pm.loadPluginByName(pluginName), PluginManager::PluginLoadStatus::SUCCESS );
        const std::string pluginPath = pm.findPlugin(pluginName);
        ASSERT_GT(pluginPath.size(), 0u);
    }

    /// Check that non existing plugin are currectly handled and returns an
    /// error message.
    {
        EXPECT_MSG_NOEMIT(Warning);
        EXPECT_MSG_EMIT(Error);
        ASSERT_EQ(pm.loadPluginByName(nonpluginName), PluginManager::PluginLoadStatus::PLUGIN_FILE_NOT_FOUND);

        ASSERT_EQ(pm.findPlugin(nonpluginName).size(), 0u);
    }
}

TEST_F(PluginManager_test, pluginEntries)
{
    PluginManager&pm = PluginManager::getInstance();

    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);
    const sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    EXPECT_TRUE(p.initExternalModule.func != nullptr);
    EXPECT_TRUE(p.getModuleName.func != nullptr);
    EXPECT_TRUE(p.getModuleVersion.func != nullptr);
    EXPECT_TRUE(p.getModuleLicense.func != nullptr);
    EXPECT_TRUE(p.getModuleDescription.func != nullptr);
    EXPECT_TRUE(p.getModuleComponentList.func != nullptr);

}

TEST_F(PluginManager_test, pluginEntriesValues)
{
    PluginManager&pm = PluginManager::getInstance();

    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);
    sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    std::string testModuleName = "TestPlugin";
    std::string testModuleVersion = "0.7";
    std::string testModuleLicense = "LicenseTest";
    std::string testModuleDescription = "Description of the Test Plugin";
    std::string testModuleComponentList = "ComponentA, ComponentB";

    ASSERT_EQ(0, std::string(p.getModuleName()).compare(testModuleName));
    ASSERT_NE(0, std::string(p.getModuleName()).compare(testModuleName + "azerty"));

    ASSERT_EQ(0, std::string(p.getModuleVersion()).compare(testModuleVersion));
    ASSERT_NE(0, std::string(p.getModuleVersion()).compare(testModuleVersion + "77777"));

    ASSERT_EQ(0, std::string(p.getModuleLicense()).compare(testModuleLicense));
    ASSERT_NE(0, std::string(p.getModuleLicense()).compare(testModuleLicense + "GPLBSDProprio"));

    ASSERT_EQ(0, std::string(p.getModuleDescription()).compare(testModuleDescription));
    ASSERT_NE(0, std::string(p.getModuleDescription()).compare(testModuleDescription + "blablablabalbal"));

    ASSERT_EQ(0, std::string(p.getModuleComponentList()).compare(testModuleComponentList));
    ASSERT_NE(0, std::string(p.getModuleComponentList()).compare(testModuleComponentList + "ComponentZ"));

}

TEST_F(PluginManager_test, testIniFile)
{
    EXPECT_MSG_NOEMIT(Deprecated);

    PluginManager&pm = PluginManager::getInstance();
    pm.loadPluginByName(pluginName);
    const std::string pluginPath = pm.findPlugin(pluginName);

    const std::string pathIniFile = "PluginManager_test.ini";
    pm.writeToIniFile(pathIniFile);

    //writeToIniFile does not return anything to say if the file was created without error...
    ASSERT_TRUE(FileSystem::exists(pathIniFile));

    createdFilesToDelete.push_back(pathIniFile);

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    ASSERT_TRUE(FileSystem::exists(pathIniFile));

    pm.readFromIniFile(pathIniFile);
    ASSERT_EQ(pm.findPlugin(pluginName).compare(pluginPath), 0);

}

TEST_F(PluginManager_test, testDeprecatedIniFileWoVersion)
{
    EXPECT_MSG_EMIT(Deprecated);

    PluginManager&pm = PluginManager::getInstance();

    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    const std::string pathIniFile = "PluginManager_test_deprecated_wo_version.ini";
    std::ofstream outstream(pathIniFile.c_str());

    outstream << pluginName << std::endl;
    outstream.close();
    ASSERT_TRUE(FileSystem::exists(pathIniFile));
    pm.readFromIniFile(pathIniFile);

    createdFilesToDelete.push_back(pathIniFile);
}
