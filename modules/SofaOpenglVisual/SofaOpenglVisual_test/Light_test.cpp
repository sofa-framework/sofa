#include <vector>
using std::vector;

#include <string>
using std::string;

#include <gtest/gtest.h>

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::Message ;
using sofa::helper::logging::MessageAsTestFailure ;

namespace light_test
{

int initMessage(){
    //MessageDispatcher::clearHandlers() ;
    //MessageDispatcher::addHandler(new ClangMessageHandler()) ;
    return 0;
}

int messageInited = initMessage();

struct TestLight : public ::testing::Test {
    void checkSpotLightValidAttributes();
    void checkDirectionalLightValidAttributes();
    void checkPositionalLightValidAttributes();
    void checkLightMissingLightManager(const std::string& lighttype);
};

void TestLight::checkLightMissingLightManager(const std::string& lighttype)
{
    MessageAsTestFailure error(Message::Error) ;
    ExpectMessage warning(Message::Warning) ;

    if(sofa::simulation::getSimulation()==nullptr)
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <"<< lighttype << " name='light1'/>                                            \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(lm, nullptr) ;

    sofa::simulation::getSimulation()->unload(root);

}

void TestLight::checkPositionalLightValidAttributes()
{
    MessageAsTestFailure warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    if(sofa::simulation::getSimulation()==nullptr)
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <PositionalLight name='light1'/>                                            \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {///These are the attributes that any light must have.
                                "drawSource", "zNear", "zFar",
                                "shadowsEnabled", "softShadows", "textureUnit",

                                /// These are the attribute specific to this type of light
                                "fixed", "position", "attenuation"
                               };

    for(auto& attrname : attrnames)
        EXPECT_NE( light->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::getSimulation()->unload(root);

}

void TestLight::checkDirectionalLightValidAttributes()
{
    MessageAsTestFailure warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    if(sofa::simulation::getSimulation()==nullptr)
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <DirectionalLight name='light1'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {///These are the attributes that any light must have.
                                "drawSource", "zNear", "zFar",
                                "shadowsEnabled", "softShadows", "textureUnit",

                                /// These are the attribute specific to this type of light
                                "direction"
                               };

    for(auto& attrname : attrnames)
        EXPECT_NE( light->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::getSimulation()->unload(root);

}

void TestLight::checkSpotLightValidAttributes()
{
    MessageAsTestFailure warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    if(sofa::simulation::getSimulation()==nullptr)
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LightManager name='lightmanager'/>                                         \n"
             "   <SpotLight name='light1'/>                                                  \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    ASSERT_NE(lm, nullptr) ;

    BaseObject* light = root->getTreeNode("Level 1")->getObject("light1") ;
    ASSERT_NE(light, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {///These are the attributes that any light must have.
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

    sofa::simulation::getSimulation()->unload(root);

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

TEST_F(TestLight, checkPositionalLightMissingLightManager)
{
    std::vector<std::string> typeoflight={"PositionalLight", "DirectionalLight", "SpotLight"} ;
    for(auto& lighttype : typeoflight)
        checkLightMissingLightManager(lighttype);
}
} // light_test
