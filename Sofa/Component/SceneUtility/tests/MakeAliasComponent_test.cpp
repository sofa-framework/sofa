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
#include <gtest/gtest.h>

#include <string>
using std::string ;

#include <sofa/testing/BaseTest.h>
#include <sofa/testing/TestMessageHandler.h>

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/component/sceneutility/MakeAliasComponent.h>
using sofa::component::sceneutility::MakeAliasComponent ;

//TODO(dmarchal): all these lines are ugly...this is too much for simple initialization stuff.
#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::MessageDispatcher;
using sofa::helper::logging::MessageHandler;
using sofa::helper::logging::ConsoleMessageHandler;
using sofa::helper::logging::Message ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter ;

using sofa::core::objectmodel::ComponentState ;

#include <sofa/simulation/graph/SimpleApi.h>

//TODO(dmarchal): handle properly the memory cycle of the simulation objects.
// now it is soo ugly...

namespace makealiascomponent_test
{

MessageHandler* defaultHandler=nullptr;
Simulation* theSimulation = nullptr ;

bool doInit(){
    return true;
}
bool inited = doInit();

void perTestInit()
{
    sofa::simpleapi::importPlugin("Sofa.Component.SceneUtility");

    theSimulation = sofa::simulation::getSimulation();

    if(defaultHandler==nullptr)
        defaultHandler=new ConsoleMessageHandler(&RichConsoleStyleMessageFormatter::getInstance()) ;

    /// THE TESTS HERE ARE NOT INHERITING FROM SOFA TEST SO WE NEED TO MANUALLY INSTALL THE HANDLER
    /// DO NO REMOVE
    MessageDispatcher::addHandler( sofa::testing::MainGtestMessageHandler::getInstance() );
}


TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias/>                                                 "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;

    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    sofa::simulation::node::unload(root);
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingTargetAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias                            alias='NewName'/>      "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    sofa::simulation::node::unload(root);
}

TEST(MakeAliasComponent, checkGracefullHandlingOfMissingAliasAttributes)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;

    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MakeAlias targetcomponent='MakeAlias'/>                     "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    sofa::simulation::node::unload(root);
}

TEST(MakeAliasComponent, checkGracefullHandlingOfInvalidTargetName)
{
    perTestInit();
    EXPECT_MSG_EMIT(Error) ;


    const string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='InvalidComponentName' alias='Something'/> \n"
        "</Node>                                                             \n" ;


    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Invalid) ;
    sofa::simulation::node::unload(root);
}

TEST(MakeAliasComponent, checkValidBehavior)
{
    EXPECT_MSG_NOEMIT(Error) ;

    const string scene =
        "<?xml version='1.0'?>                                               \n"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         \n"
        "       <MakeAlias targetcomponent='MakeAlias' alias='MyAlias'/>     \n"
        "       <MyAlias targetcomponent='MakeAlias' alias='ThirdName'/>     \n"
        "</Node>                                                             \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MakeAliasComponent* component = nullptr;
    root->getTreeObject(component) ;

    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_EQ(component->getComponentState(), ComponentState::Valid) ;
    sofa::simulation::node::unload(root);
}

}
