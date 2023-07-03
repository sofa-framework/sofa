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

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::execparams::defaultInstance; 

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

#include <sofa/simulation/graph/SimpleApi.h>

namespace cliplane_test
{

int initMessage(){
    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestClipPlane : public BaseTest {
public:
    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.GL.Component.Rendering3D");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
    }

    void checkClipPlaneValidAttributes();
    void checkClipPlaneAttributesValues(const std::string& dataname, const std::string& value);
};

void TestClipPlane::checkClipPlaneValidAttributes()
{
    EXPECT_MSG_NOEMIT(Warning, Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <ClipPlane name='clipplane'/>                                               \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {"position", "normal", "id", "active"};

    for(auto& attrname : attrnames)
        EXPECT_NE( clp->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}


void TestClipPlane::checkClipPlaneAttributesValues(const std::string& dataname, const std::string& value)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <ClipPlane name='clipplane'"
          << dataname << "='" << value <<
             "'/>                                                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}

TEST_F(TestClipPlane, checkClipPlaneIdInValidValues)
{
    EXPECT_MSG_EMIT(Error) ;

    checkClipPlaneAttributesValues("id", "-1");
}

TEST_F(TestClipPlane, checkClipPlaneNormalInvalidNormalValue)
{
    EXPECT_MSG_EMIT(Warning) ;

    checkClipPlaneAttributesValues("normal", "1 0");
}


} // cliplane_test
