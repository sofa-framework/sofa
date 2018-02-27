/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
using sofa::helper::Console ;

#include <sofa/helper/testing/TestMessageHandler.h>
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::MainGtestMessageHandler ;

#include <sofa/helper/random.h>

#include "BaseTest.h"

namespace sofa {
namespace helper {
namespace testing {

void initializeOnce()
{
    static bool initialized = false ;
    if(!initialized){
        Console::setColorsStatus(Console::ColorsDisabled) ;

        MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
        BackTrace::autodump() ;

        initialized=true ;
    }
}

int BaseTest::seed = (unsigned int)time(NULL);

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
    Console::setColorsStatus(Console::ColorsDisabled) ;

    ///Repeating this for each class is harmless because addHandler test if the handler is already installed and
    ///if so it don't install it again.
    MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
}

BaseTest::~BaseTest() {}

void BaseTest::SetUp()
{
    onSetUp();
}

void BaseTest::TearDown()
{
    onTearDown();
}


} /// testing
} /// helper
} /// sofa


