/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <fstream>
#include <gtest/gtest.h>
#include <SofaTest/Sofa_test.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

using sofa::helper::system::DataRepository;
using sofa::helper::system::PluginRepository;
using sofa::helper::system::PluginManager;

class runSofa_test : public Sofa_test<>
{
protected:
    std::string m_testConfigPluginName;
    std::string m_testConfigPluginPath;
    std::string m_testPluginName;

    runSofa_test() {

    }

    void SetUp()
    {
        const std::string& pluginDir = PluginRepository.getFirstPath();

        m_testConfigPluginName = "test_plugin_list.conf";
        m_testConfigPluginPath = pluginDir + "/" + m_testConfigPluginName;
        m_testPluginName = "TestPlugin";
        
        //generate on the fly test list
        std::ofstream testPluginList;
        testPluginList.open(m_testConfigPluginPath);
        testPluginList << m_testPluginName << std::endl;
        testPluginList.close();
    }
    void TearDown()
    {

    }
    
};

TEST_F(runSofa_test, runSofa_autoload)
{
    PluginManager& pm = PluginManager::getInstance();
    unsigned int num = pm.getPluginMap().size() ;
    pm.readFromIniFile(m_testConfigPluginPath);
    PluginManager::getInstance().init();
    ASSERT_GT(pm.getPluginMap().size(), num);
    const std::string pluginPath = pm.findPlugin(m_testPluginName);
    ASSERT_GT(pluginPath.size(), 0U);
    helper::system::Plugin& p = pm.getPluginMap()[pluginPath];
    ASSERT_EQ(0, std::string(p.getModuleName()).compare(m_testPluginName));
}

} // namespace sofa
