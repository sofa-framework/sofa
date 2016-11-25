#include <vector>
using std::vector;

#include <string>
using std::string;

#include <SofaTest/Sofa_test.h>

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
using sofa::helper::logging::MessageAsTestFailure;
using sofa::helper::logging::Message ;

int initMessage(){
    //MessageDispatcher::clearHandlers() ;
    //MessageDispatcher::addHandler(new ClangMessageHandler()) ;
    return 0;
}
int messageInited = initMessage();

namespace sofa {

struct TestLightManager : public ::testing::Test {
    void checkAttributes();
    void checkColor_Ambient_OK();
    void checkColor_Ambient_OpenIssue();
};

void  TestLightManager::checkAttributes()
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Vec3d'/>                                        \n"
             "   <LightManager name='lightmanager'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    EXPECT_NE(lm, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {
        "shadows", "softShadows", "ambient", "debugDraw"
    };

    for(auto& attrname : attrnames)
        EXPECT_NE( lm->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;
}

void TestLightManager::checkColor_Ambient_OK()
{
    MessageAsTestFailure error(Message::Error) ;
    MessageAsTestFailure warning(Message::Warning) ;

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Vec3d'/>                                        \n"
             "   <LightManager name='lightmanager' ambient='1 0 2 3'/>                         \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    EXPECT_NE(lm, nullptr) ;
}

void TestLightManager::checkColor_Ambient_OpenIssue()
{
    MessageAsTestFailure error(Message::Error) ;
    MessageAsTestFailure warning(Message::Warning) ;

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Vec3d'/>                                        \n"
             "   <LightManager name='lightmanager' ambient='red'/>                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lm = root->getTreeNode("Level 1")->getObject("lightmanager") ;
    EXPECT_NE(lm, nullptr) ;
}

TEST_F(TestLightManager, checkAttributes)
{
    checkAttributes();
}

TEST_F(TestLightManager, checkColor_Ambient_OK)
{
    checkColor_Ambient_OK();
}
TEST_F(TestLightManager, checkColor_Ambient_OpenIssue)
{
    checkColor_Ambient_OpenIssue();
}

}
