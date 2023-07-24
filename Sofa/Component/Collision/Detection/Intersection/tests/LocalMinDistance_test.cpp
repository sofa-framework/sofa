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

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <sofa/component/collision/detection/intersection/LocalMinDistance.h>
using sofa::component::collision::detection::intersection::LocalMinDistance ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/simulation/graph/SimpleApi.h>

#include <gtest/gtest-spi.h> // for expected non fatal

using namespace sofa::testing;
using namespace sofa::helper::logging;

namespace
{

struct TestLocalMinDistance : public BaseSimulationTest {
    void SetUp()
    {
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
    }
    void TearDown()
    {
    }

    void checkAttributes();
    void checkMissingRequiredAttributes();

    void checkDoubleInit();
    void checkInitReinit();

    void checkBasicIntersectionTests();
    void checkIfThereIsAnExampleFile();
};

void TestLocalMinDistance::checkBasicIntersectionTests()
{
    ExpectMessage warning(Message::Warning) ;

    assert(sofa::simulation::getSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    auto* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    LocalMinDistance* lmdt = dynamic_cast<LocalMinDistance*>(lmd);
    ASSERT_NE(lmdt, nullptr) ;

    sofa::simulation::node::unload(root);
}


void TestLocalMinDistance::checkMissingRequiredAttributes()
{
    ExpectMessage warning(Message::Warning) ;

    assert(sofa::simulation::getSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    auto* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    sofa::simulation::node::unload(root);
}

void TestLocalMinDistance::checkAttributes()
{
    assert(sofa::simulation::getSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    auto* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {
        "filterIntersection", "angleCone",   "coneFactor", "useLMDFilters"
    };

    for(auto& attrname : attrnames)
        EXPECT_NE( lmd->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::node::unload(root);
}


void TestLocalMinDistance::checkDoubleInit()
{
    assert(sofa::simulation::getSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    auto* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    lmd->init() ;

    FAIL() << "TODO: Calling init twice does not produce any warning message";
 
    sofa::simulation::node::unload(root);
}


void TestLocalMinDistance::checkInitReinit()
{
    assert(sofa::simulation::getSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    auto* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    lmd->reinit() ;

    sofa::simulation::node::unload(root);
}


TEST_F(TestLocalMinDistance, checkAttributes)
{
    checkAttributes();
}

TEST_F(TestLocalMinDistance, checkBasicIntersectionTests_OpenIssue)
{
    checkBasicIntersectionTests();
}

TEST_F(TestLocalMinDistance, DISABLED_checkDoubleInit_OpenIssue)
{
    checkDoubleInit();
}

TEST_F(TestLocalMinDistance, checkMissingRequiredAttributes)
{
    checkMissingRequiredAttributes();
}



}
