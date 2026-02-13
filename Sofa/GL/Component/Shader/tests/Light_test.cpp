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
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::execparams::defaultInstance; 

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simpleapi/SimpleApi.h>

namespace light_test
{

int initMessage(){
    static sofa::gl::component::rendering3d::OglModel* t;

    if(t){
        t->init();
    }

    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestLight : public BaseTest {
public:

    void doSetUp() override
    {
        this->loadPlugins({
            Sofa.GL.Component.Shader,
            Sofa.Component.StateContainer
        });
    }

    void checkSpotLightValidAttributes();
    void checkDirectionalLightValidAttributes();
    void checkPositionalLightValidAttributes();
    void checkLightMissingLightManager(const std::string& lighttype);
};

void TestLight::checkLightMissingLightManager(const std::string& lighttype)
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <"<< lighttype << " name='light1'/>                                            \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    sofa::core::objectmodel::BaseObject* lm = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(lm, nullptr) ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}

void TestLight::checkPositionalLightValidAttributes()
{
    EXPECT_MSG_NOEMIT(Warning, Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <PositionalLight name='light1'/>                                            \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    sofa::core::objectmodel::BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    sofa::core::objectmodel::BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {///These are the attributes that any light must have.
                                "drawSource", "zNear", "zFar",
                                "shadowsEnabled", "softShadows", "textureUnit",

                                /// These are the attribute specific to this type of light
                                "fixed", "position", "attenuation"
                               };

    for(auto& attrname : attrnames)
        EXPECT_NE( light->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}

void TestLight::checkDirectionalLightValidAttributes()
{
    EXPECT_MSG_NOEMIT(Warning, Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <DirectionalLight name='light1'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    sofa::core::objectmodel::BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    sofa::core::objectmodel::BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {///These are the attributes that any light must have.
                                "drawSource", "zNear", "zFar",
                                "shadowsEnabled", "softShadows", "textureUnit",

                                /// These are the attribute specific to this type of light
                                "direction"
                               };

    for(auto& attrname : attrnames)
        EXPECT_NE( light->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}

void TestLight::checkSpotLightValidAttributes()
{
    EXPECT_MSG_NOEMIT(Warning, Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <SpotLight name='light1'/>                                                  \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    sofa::core::objectmodel::BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    sofa::core::objectmodel::BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {///These are the attributes that any light must have.
                                "drawSource", "zNear", "zFar",
                                "shadowsEnabled", "softShadows", "textureUnit",

                                /// These are the attributes inherited from PositionalLight
                                 "fixed", "position", "attenuation",

                                /// These are the attribute specific to this type of light
                                "direction", "cutoff", "exponent", "lookat",
                                "modelViewMatrix", "projectionMatrix"
                               };

    for(auto& attrname : attrnames)
        EXPECT_NE( light->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::node::unload(root);
    sofa::simulation::getSimulation()->createNewGraph("");
}


TEST_F(TestLight, checkPositionalLightValidAttributes)
{
    checkPositionalLightValidAttributes();
}

TEST_F(TestLight, checkDirectionalLightValidAttributes)
{
    checkDirectionalLightValidAttributes();
}

TEST_F(TestLight, checkSpotLightValidAttributes)
{
    checkSpotLightValidAttributes();
}

TEST_F(TestLight, checkLightMissingLightManager)
{
    const std::vector<std::string> typeoflight={"PositionalLight", "DirectionalLight", "SpotLight"} ;
    for(auto& lighttype : typeoflight)
        checkLightMissingLightManager(lighttype);
}
} // light_test
