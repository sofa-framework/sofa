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

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/component/sceneutility/MessageHandlerComponent.h>
using sofa::component::sceneutility::MessageHandlerComponent ;

using sofa::helper::logging::MessageDispatcher ;

#include <sofa/simulation/graph/SimpleApi.h>

bool perTestInit()
{
    sofa::simpleapi::importPlugin("Sofa.Component.SceneUtility");

    /// THE TESTS HERE ARE NOT INHERITING FROM SOFA TEST SO WE NEED TO MANUALLY INSTALL THE HANDLER
    /// DO NO REMOVE
    MessageDispatcher::addHandler( sofa::testing::MainGtestMessageHandler::getInstance() );
    return true;
}
bool inited = perTestInit() ;



TEST(MessageHandlerComponent, simpleInit)
{
    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "   <Node>  "
            "       <MessageHandlerComponent handler='silent'/>              "
        "   </Node> "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());
    EXPECT_TRUE(root!=nullptr) ;

    MessageHandlerComponent* component = nullptr;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
}


TEST(MessageHandlerComponent, missingHandler)
{
    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent/>                   "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());

    MessageHandlerComponent* component = nullptr;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_FALSE(component->isValid()) ;
}

TEST(MessageHandlerComponent, invalidHandler)
{
    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent handler='thisisinvalid'/>           "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());

    MessageHandlerComponent* component = nullptr;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_FALSE(component->isValid()) ;
}

TEST(MessageHandlerComponent, clangHandler)
{
    const string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent handler='clang'/>                   "
        "</Node>                                                             " ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("test1", scene.c_str());

    MessageHandlerComponent* component = nullptr;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=nullptr) ;
    EXPECT_TRUE(component->isValid()) ;
}
