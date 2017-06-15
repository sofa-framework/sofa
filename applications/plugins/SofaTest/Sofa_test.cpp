/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include "Sofa_test.h"
#include <SceneCreator/SceneCreator.h>

#include <sofa/helper/system/FileRepository.h>
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

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::MainGtestMessageHandler ;

namespace sofa {

// some basic RAII stuff to automatically add a TestMessageHandler to every tests
namespace {
    static struct raii {
      raii() {
          MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
          BackTrace::autodump() ;
      }
    } singleton;
}


int BaseSofa_test::seed = (unsigned int)time(NULL);

BaseSofa_test::BaseSofa_test(){
    seed = testing::UnitTest::GetInstance()->random_seed() ;
    modeling::initSofa();

    //if you want to generate the same sequence of pseudo-random numbers than a specific test suites
    //use the same seed (the seed value is indicated at the 2nd line of test results)
    //and pass the seed in command argument line ex: SofaTest_test.exe seed 32
    helper::srand(seed);

    // gtest already use color so we remove the color from the sofa message to make the distinction
    // clean and avoid ambiguity.
    Console::setColorsStatus(Console::ColorsDisabled) ;

    // Repeating this for each class is harmless because addHandler test if the handler is already installed and
    // if so it don't install it again.
    MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() ) ;
}

BaseSofa_test::~BaseSofa_test(){ clearSceneGraph(); }

void BaseSofa_test::clearSceneGraph(){ modeling::clearScene(); }


#ifndef SOFA_FLOAT
template struct SOFA_TestPlugin_API Sofa_test<double>;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_TestPlugin_API Sofa_test<float>;
#endif
}
