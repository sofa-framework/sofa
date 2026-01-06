/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/BaseTest.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::PluginRepository ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::PluginRepository;
using sofa::helper::system::DataRepository;
using sofa::helper::system::FileSystem;

#include <sofa/helper/Utils.h>
using sofa::helper::Utils;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace;

#include <sofa/helper/system/console.h>


#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/logging/MessageDispatcher.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/testing/TestMessageHandler.h>
using sofa::testing::MainGtestMessageHandler ;

#include <sofa/helper/random.h>


namespace sofa::testing
{

void initializeOnce()
{
    static bool initialized = false ;
    if(!initialized){
        helper::console::setStatus(helper::console::Status::Off) ;

        MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
        BackTrace::autodump() ;

        initialized=true ;
    }
}

int BaseTest::seed = (unsigned int)time(nullptr);

BaseTest::BaseTest() :
   m_fatal(sofa::helper::logging::Message::Fatal, __FILE__, __LINE__ ),
   m_error(sofa::helper::logging::Message::Error, __FILE__, __LINE__ )
{
    initializeOnce() ;

    seed = ::testing::UnitTest::GetInstance()->random_seed() ;

    ///if you want to generate the same sequence of pseudo-random numbers than a specific test suites
    ///use the same seed (the seed value is indicated at the 2nd line of test results)
    ///and pass the seed in command argument line ex: SofaTest_test.exe seed 32
    helper::srand(seed);

    ///gtest already use color so we remove the color from the sofa message to make the distinction
    ///clean and avoid ambiguity.
    helper::console::setStatus(helper::console::Status::Off);

    ///Repeating this for each class is harmless because addHandler test if the handler is already installed and
    ///if so it don't install it again.
    MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
}

BaseTest::~BaseTest() {}

void BaseTest::loadPlugins(
    const std::initializer_list<std::string>& pluginNames)
{
    m_loadedPlugins.emplace_back(pluginNames.begin(), pluginNames.end());
}

void BaseTest::SetUp()
{
    doSetUp();
}

void BaseTest::TearDown()
{
    m_loadedPlugins.clear();
    doTearDown();
}

} // namespace sofa::testing
