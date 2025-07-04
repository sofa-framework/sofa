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
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <sofa/simulation/Node.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <fstream>

using sofa::helper::system::PluginManager;
using sofa::helper::system::FileSystem;

static std::string pluginAName = "TestPluginA";
static std::string pluginBName = "TestPluginB";
static std::string failingPluginName = "FailingPlugin";

#ifdef NDEBUG
static std::string pluginAFileName = "TestPluginA";
static std::string pluginBFileName = "TestPluginB";
#else
static std::string pluginAFileName = "TestPluginA_d";
static std::string pluginBFileName = "TestPluginB_d";
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

    void doSetUp() override
    {
        // Set pluginDir by searching pluginFileName in the PluginRepository
        for ( std::string path : sofa::helper::system::PluginRepository.getPaths() )
        {
            if ( FileSystem::exists(path + separator + prefix + pluginAFileName + dotExt) )
            {
                pluginDir = path;
                break;
            }
        }

        ASSERT_FALSE( pluginDir.empty() );

        std::cout << "PluginManager_test.loadTestPluginAByPath: "
                  << "pluginDir = " << pluginDir
                  << std::endl;
    }

    void doTearDown() override
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


TEST_F(PluginManager_test, loadTestPluginAByPath)
{
    PluginManager&pm = PluginManager::getInstance();

    const std::string pluginPath = pluginDir + separator + prefix + pluginAFileName + dotExt;
    const std::string nonpluginPath = pluginDir + separator + prefix + nonpluginName + dotExt;

    std::cout << "PluginManager_test.loadTestPluginAByPath: "
              << "pluginPath = " << pluginPath
              << std::endl;

    /// Check that existing plugins are correctly handled and returns no
    /// error/warning message.
    {
        EXPECT_MSG_NOEMIT(Error, Warning);
        
        std::cout << "PluginManager_test.loadTestPluginAByPath: "
        << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
        << std::endl;
    }
    {
        EXPECT_MSG_NOEMIT(Error);
        // Plugin A still uses the deprecated registration mechanism
        // and is expected to throw a warning when loaded
        EXPECT_MSG_EMIT(Warning);
        
        ASSERT_EQ(pm.loadPluginByPath(pluginPath), PluginManager::PluginLoadStatus::SUCCESS);
    }
    {
        EXPECT_MSG_NOEMIT(Error, Warning);
        ASSERT_GT(pm.findPlugin(pluginAName).size(), 0u);
    }

    /// Check that non existing plugin are currectly handled and returns an
    /// error message.
    {
        EXPECT_MSG_NOEMIT(Warning);
        EXPECT_MSG_EMIT(Error);

        std::cout << "PluginManager_test.loadTestPluginAByPath: "
                  << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
                  << std::endl;
        ASSERT_EQ(pm.loadPluginByPath(nonpluginPath), PluginManager::PluginLoadStatus::PLUGIN_FILE_NOT_FOUND);
        ASSERT_EQ(pm.findPlugin(nonpluginName).size(), 0u);
        std::cout << "PluginManager_test.loadTestPluginAByPath: "
                  << "pm.getPluginMap().size() = " << pm.getPluginMap().size()
                  << std::endl;
    }
}

TEST_F(PluginManager_test, loadTestPluginAByName )
{
    PluginManager&pm = PluginManager::getInstance();

    /// Check that existing plugins are correctly handled and returns no
    /// error/warning message.
    {
        EXPECT_MSG_NOEMIT(Warning, Error);

        ASSERT_EQ(pm.loadPluginByName(pluginAName), PluginManager::PluginLoadStatus::SUCCESS );
        const std::string pluginPath = pm.findPlugin(pluginAName);
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

    pm.loadPluginByName(pluginAName);
    const std::string pluginPath = pm.findPlugin(pluginAName);
    const sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    EXPECT_TRUE(p.initExternalModule.func != nullptr);
    EXPECT_TRUE(p.getModuleName.func != nullptr);
    EXPECT_TRUE(p.getModuleVersion.func != nullptr);
    EXPECT_TRUE(p.getModuleLicense.func != nullptr);
    EXPECT_TRUE(p.getModuleDescription.func != nullptr);
}

TEST_F(PluginManager_test, pluginEntriesValues)
{
    PluginManager&pm = PluginManager::getInstance();

    pm.loadPluginByName(pluginAName);
    const std::string pluginPath = pm.findPlugin(pluginAName);
    sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    std::string testModuleName = "TestPluginA";
    std::string testModuleVersion = "0.7";
    std::string testModuleLicense = "LicenseTest";
    std::string testModuleDescription = "Description of the Test Plugin A";
    std::string testModuleComponentList = "ComponentA, ComponentB";

    ASSERT_EQ(0, std::string(p.getModuleName()).compare(testModuleName));
    ASSERT_NE(0, std::string(p.getModuleName()).compare(testModuleName + "azerty"));

    ASSERT_EQ(0, std::string(p.getModuleVersion()).compare(testModuleVersion));
    ASSERT_NE(0, std::string(p.getModuleVersion()).compare(testModuleVersion + "77777"));

    ASSERT_EQ(0, std::string(p.getModuleLicense()).compare(testModuleLicense));
    ASSERT_NE(0, std::string(p.getModuleLicense()).compare(testModuleLicense + "GPLBSDProprio"));

    ASSERT_EQ(0, std::string(p.getModuleDescription()).compare(testModuleDescription));
    ASSERT_NE(0, std::string(p.getModuleDescription()).compare(testModuleDescription + "blablablabalbal"));

    ASSERT_EQ(0, std::string(sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(testModuleName)).compare(testModuleComponentList));
    ASSERT_NE(0, std::string(sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(testModuleName)).compare(testModuleComponentList + "ComponentZ"));

}

TEST_F(PluginManager_test, testIniFile)
{
    EXPECT_MSG_NOEMIT(Deprecated);

    PluginManager&pm = PluginManager::getInstance();
    pm.loadPluginByName(pluginAName);
    const std::string pluginPath = pm.findPlugin(pluginAName);

    const std::string pathIniFile = "PluginManager_test.ini";
    pm.writeToIniFile(pathIniFile);

    //writeToIniFile does not return anything to say if the file was created without error...
    ASSERT_TRUE(FileSystem::exists(pathIniFile));

    createdFilesToDelete.push_back(pathIniFile);

    ASSERT_TRUE(pm.unloadPlugin(pluginPath));
    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    ASSERT_TRUE(FileSystem::exists(pathIniFile));

    pm.readFromIniFile(pathIniFile);
    ASSERT_EQ(pm.findPlugin(pluginAName).compare(pluginPath), 0);

}

TEST_F(PluginManager_test, testDeprecatedIniFileWoVersion)
{
    EXPECT_MSG_EMIT(Deprecated);

    PluginManager&pm = PluginManager::getInstance();

    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    const std::string pathIniFile = "PluginManager_test_deprecated_wo_version.ini";
    std::ofstream outstream(pathIniFile.c_str());

    outstream << pluginAName << std::endl;
    outstream.close();
    ASSERT_TRUE(FileSystem::exists(pathIniFile));
    pm.readFromIniFile(pathIniFile);

    createdFilesToDelete.push_back(pathIniFile);
}

TEST_F(PluginManager_test, testPluginAAsDependencyOfPluginB)
{
    std::string testModuleName = "TestPluginB";
    PluginManager&pm = PluginManager::getInstance();

    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    pm.loadPluginByName(pluginBName);
    const std::string pluginPath = pm.findPlugin(pluginBName);
    const sofa::helper::system::Plugin& p = pm.getPluginMap()[pluginPath];

    EXPECT_TRUE(p.initExternalModule.func != nullptr);
    EXPECT_TRUE(p.getModuleName.func != nullptr);
    ASSERT_EQ(0, std::string(p.getModuleName()).compare(testModuleName));

    // TestPluginB does not implement the other get*() functions
    EXPECT_FALSE(p.getModuleVersion.func != nullptr);
    EXPECT_FALSE(p.getModuleLicense.func != nullptr);
    EXPECT_FALSE(p.getModuleDescription.func != nullptr);
    EXPECT_FALSE(p.getModuleComponentList.func != nullptr);

}


TEST_F(PluginManager_test, failingPlugin)
{
    std::string testModuleName = "FailingPlugin";
    PluginManager&pm = PluginManager::getInstance();

    ASSERT_EQ(pm.getPluginMap().size(), 0u);

    EXPECT_EQ(pm.unloadedPlugins().find(failingPluginName), pm.unloadedPlugins().end());
    EXPECT_FALSE(sofa::core::ObjectFactory::getInstance()->hasCreator("ComponentFailingPlugin"));
    {
        EXPECT_MSG_EMIT(Error); //because initialization will fail
        pm.loadPluginByName(failingPluginName);
    }
    EXPECT_NE(pm.unloadedPlugins().find(failingPluginName), pm.unloadedPlugins().end());

    const std::string pluginPath = pm.findPlugin(failingPluginName);
    EXPECT_EQ(pm.getPluginMap().find(pluginPath), pm.getPluginMap().end());

    EXPECT_TRUE(sofa::core::ObjectFactory::getInstance()->hasCreator("ComponentFailingPlugin"));

    sofa::core::objectmodel::BaseObjectDescription description("ComponentFailingPlugin", "ComponentFailingPlugin");
    const auto tmpNode = sofa::core::objectmodel::New<sofa::simulation::Node>("tmp");
    EXPECT_EQ(sofa::core::ObjectFactory::getInstance()->createObject(tmpNode.get(), &description), nullptr);

    EXPECT_FALSE(description.getErrors().empty());
    EXPECT_NE(std::find_if(description.getErrors().begin(), description.getErrors().end(),
        [](const std::string& error)
        {
            return error.find(
                "The object was previously registered, but the module that "
                "registered the object has been unloaded, preventing the object creation.")
            != std::string::npos;
        }), description.getErrors().end());

    std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries, false);
    EXPECT_NE(
        std::find_if(entries.begin(), entries.end(), [](const auto& entry){ return entry->className == "ComponentFailingPlugin";}),
        entries.end()
    );

    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries, true);
    EXPECT_EQ(
        std::find_if(entries.begin(), entries.end(), [](const auto& entry){ return entry->className == "ComponentFailingPlugin";}),
        entries.end()
    );
}
