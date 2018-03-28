/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <Sofa.Component.Utils.h>

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

#include <sofa/component/utils/MessageHandlerComponent.h>
using sofa::component::utils::logging::MessageHandlerComponent ;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::MainGtestMessageHandler ;
using sofa::helper::logging::MessageDispatcher ;

bool perTestInit()
{
    /// THE TESTS HERE ARE NOT INHERITING FROM SOFA TEST SO WE NEED TO MANUALLY INSTALL THE HANDLER
    /// DO NO REMOVE
    MessageDispatcher::addHandler( MainGtestMessageHandler::getInstance() );
    return true;
}
bool inited = perTestInit() ;



TEST(MessageHandlerComponent, simpleInit)
{
    sofa::component::utils::initSofaComponentUtils();

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "   <Node>  "
            "       <MessageHandlerComponent handler='silent'/>              "
        "   </Node> "
        "</Node>                                                             " ;

    sofa::simulation::setSimulation(new DAGSimulation());

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;
    EXPECT_TRUE(root!=NULL) ;

    MessageHandlerComponent* component = NULL;

    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=NULL) ;
}


TEST(MessageHandlerComponent, missingHandler)
{
    sofa::component::utils::initSofaComponentUtils();

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent/>                   "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;

    MessageHandlerComponent* component = NULL;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=NULL) ;
    EXPECT_FALSE(component->isValid()) ;
}

TEST(MessageHandlerComponent, invalidHandler)
{
    sofa::component::utils::initSofaComponentUtils();

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent handler='thisisinvalid'/>           "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;

    MessageHandlerComponent* component = NULL;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=NULL) ;
    EXPECT_FALSE(component->isValid()) ;
}

TEST(MessageHandlerComponent, clangHandler)
{
    sofa::component::utils::initSofaComponentUtils();

    string scene =
        "<?xml version='1.0'?>                                               "
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >         "
        "       <MessageHandlerComponent handler='clang'/>                   "
        "</Node>                                                             " ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ( "test1",
                                                       scene.c_str(),
                                                       scene.size() ) ;

    MessageHandlerComponent* component = NULL;
    root->getTreeObject(component) ;
    EXPECT_TRUE(component!=NULL) ;
    EXPECT_TRUE(component->isValid()) ;
}
