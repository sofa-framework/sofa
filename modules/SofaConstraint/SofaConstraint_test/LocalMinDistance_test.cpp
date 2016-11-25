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

#include <SofaConstraint/LocalMinDistance.h>
using sofa::component::collision::LocalMinDistance ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

int initMessage(){
    //MessageDispatcher::clearHandlers() ;
    //MessageDispatcher::addHandler(new ClangMessageHandler()) ;
    return 0;
}
int messageInited = initMessage();

namespace sofa {

struct TestLocalMinDistance : public ::testing::Test {
    void SetUp()
    {
        DataRepository.addFirstPath(FRAMEWORK_EXAMPLES_DIR);
    }
    void TearDown()
    {
        DataRepository.removePath(FRAMEWORK_EXAMPLES_DIR);
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

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    LocalMinDistance* lmdt = dynamic_cast<LocalMinDistance*>(lmd);
    ASSERT_NE(lmdt, nullptr) ;

    sofa::component::collision::Point p1;
    sofa::component::collision::Point p2;


    sofa::simulation::getSimulation()->unload(root);
}


void TestLocalMinDistance::checkMissingRequiredAttributes()
{
    ExpectMessage warning(Message::Warning) ;

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    sofa::simulation::getSimulation()->unload(root);
}

void TestLocalMinDistance::checkAttributes()
{
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {
        "filterIntersection", "angleCone",   "coneFactor", "useLMDFilters"
    };

    for(auto& attrname : attrnames)
        EXPECT_NE( lmd->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::getSimulation()->unload(root);
}


void TestLocalMinDistance::checkDoubleInit()
{
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    lmd->init() ;

    //TODO(dmarchal) ask consortium what is the status for double call.
    FAIL() << "TODO: Calling init twice does not produce any warning message" ;

    sofa::simulation::getSimulation()->unload(root);
}


void TestLocalMinDistance::checkInitReinit()
{
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <LocalMinDistance name='lmd'/>                                             \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* lmd = root->getTreeNode("Level 1")->getObject("lmd") ;
    ASSERT_NE(lmd, nullptr) ;

    lmd->reinit() ;

    sofa::simulation::getSimulation()->unload(root);
}


TEST_F(TestLocalMinDistance, checkAttributes)
{
    checkAttributes();
}

TEST_F(TestLocalMinDistance, checkBasicIntersectionTests_OpenIssue)
{
    checkBasicIntersectionTests();
}

//TODO(dmarchal): restore the two tests when the double call status will be clarified. deprecated after (14/11/2016)+6 month
TEST_F(TestLocalMinDistance, checkInitReinit_OpenIssue)
{
    checkInitReinit();
}

TEST_F(TestLocalMinDistance, checkDoubleInit_OpenIssue)
{
    checkDoubleInit();
}

TEST_F(TestLocalMinDistance, checkMissingRequiredAttributes)
{
    checkMissingRequiredAttributes();
}



}
