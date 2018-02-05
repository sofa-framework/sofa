/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <gtest/gtest.h>

#include <string>
using std::string ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <SofaComponentBase/initComponentBase.h>

#include <SofaComponentBase/MakeAliasComponent.h>
using sofa::component::MakeAliasComponent ;

//TODO(dmarchal): all these lines are ugly...this is too much for simple initialization stuff.
#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::MessageDispatcher;
using sofa::helper::logging::MessageHandler;
using sofa::helper::logging::ConsoleMessageHandler;
using sofa::helper::logging::Message ;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::MainGtestMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter ;

using sofa::core::objectmodel::ComponentState ;

//TODO(dmarchal): handle properly the memory cycle of the simulation objects.
// now it is soo ugly...

namespace makealiascomponent_test
{

MessageHandler* defaultHandler=nullptr;
Simulation* theSimulation = nullptr ;

bool doInit(){
    sofa::component::initComponentBase();
    return true;
}
bool inited = doInit();

void perTestInit()
{
    if(theSimulation==nullptr){
        theSimulation = new DAGSimulation();
        sofa::simulation::setSimulation(theSimulation);
    }

    if(defaultHandler==nullptr)
        defaultHandler=new ConsoleMessageHandler(new RichConsoleStyleMessageFormatter) ;

    /// THE TESTS HERE ARE NOT INHERITING FROM SOFA TEST SO WE NEED TO MANUALLY INSTALL THE HANDLER
    /// DO NO REMOVE
    MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() );
}


TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias/>                                                 "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;

    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    theSimulation->unload(root) ;
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingTargetAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias                            alias='NewName'/>      "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    theSimulation->unload(root) ;
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAliasAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias targetcomponent='MakeAlias'/>                     "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    theSimulation->unload(root) ;
}

TEST(MakeAliasComponent, checkGracefullHandlingOfInvalidTargetName)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='InvalidComponentName' alias='Something'/> \n"
        "</Node>                                                             \n" ;


    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    theSimulation->unload(root) ;
}

TEST(MakeAliasComponent, checkValidBehavior)
{
    EXPECT_MSG_NOEMIT(Error) ;

    string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='MakeAlias' alias='MyAlias'/>     \n"
        "       <MyAlias targetcomponent='MakeAlias' alias='ThirdName'/>     \n"
        "</Node>                                                             \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;
    root->getTreeObject(component) ;

    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Valid) ;
    theSimulation->unload(root) ;
}

}
