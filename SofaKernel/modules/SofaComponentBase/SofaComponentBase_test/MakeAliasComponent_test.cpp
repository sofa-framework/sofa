/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Contributors:                                                               *
*    - damien.marchal@univ-lille1.fr                                          *
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
#include <SofaTest/TestMessageHandler.h>
#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::MessageDispatcher;
using sofa::helper::logging::MessageHandler;
using sofa::helper::logging::ConsoleMessageHandler;
using sofa::helper::logging::MainCountingMessageHandler ;
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::MessageAsTestFailure ;
using sofa::helper::logging::Message ;

#include <sofa/helper/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter ;

using sofa::core::objectmodel::ComponentState ;

//TODO(dmarchal): handle properly the memory cycle of the simulation objects.
// now it is soo ugly...

MessageHandler* defaultHandler=nullptr;
bool doInit(){
    sofa::component::initComponentBase();
    return true;
}

bool inited = doInit();

void perTestInit()
{
    if(defaultHandler==nullptr)
        defaultHandler=new ConsoleMessageHandler(new RichConsoleStyleMessageFormatter) ;

    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler( &MainCountingMessageHandler::getInstance() ) ;
    MessageDispatcher::addHandler(defaultHandler);
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAttributes)
{
    perTestInit();
    ExpectMessage e(Message::Error) ;

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias/>                                                 "
        "</Node>                                                             " ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;

    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;

}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingTargetAttributes)
{
    perTestInit();
    ExpectMessage e(Message::Error) ;

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias                            alias='NewName'/>      "
        "</Node>                                                             " ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAliasAttributes)
{
    perTestInit();
    ExpectMessage e(Message::Error) ;

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias targetcomponent='MakeAlias'/>                     "
        "</Node>                                                             " ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
}

TEST(MakeAliasComponent, checkGracefullHandlingOfInvalidTargetName)
{
    perTestInit();
    ExpectMessage e(Message::Error) ;

    string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='InvalidComponentName' alias='Something'/> \n"
        "</Node>                                                             \n" ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;

}

TEST(MakeAliasComponent, checkValidBehavior)
{
    MessageAsTestFailure check(Message::Error) ;

    string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='MakeAlias' alias='MyAlias'/>     \n"
        "       <MyAlias targetcomponent='MakeAlias' alias='ThirdName'/>     \n"
        "</Node>                                                             \n" ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;
    root->getTreeObject(component) ;

    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Valid) ;

}
