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
using sofa::helper::logging::Message ;

#include <SofaConstraint/LocalMinDistance.h>
using sofa::component::collision::LocalMinDistance ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::MainCountingMessageHandler;

#include <sofa/helper/logging/LoggingMessageHandler.h>
using sofa::helper::logging::MainLoggingMessageHandler;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;


//TODO(dmarchal) to remove when the handler will be installed by sofa_test
int initMessage(){
    // We can add handler there is a check they are not duplicated in the dispatcher.
    MessageDispatcher::addHandler(&MainLoggingMessageHandler::getInstance()) ;
    MessageDispatcher::addHandler(&MainCountingMessageHandler::getInstance()) ;
    return 0;
}
int messageInited = initMessage();

namespace sofa {

struct TestLocalMinDistance : public Sofa_test<double> {
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

    clearSceneGraph();
}


void TestLocalMinDistance::checkMissingRequiredAttributes()
{
    ExpectMessage warning(Message::Warning) ;

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

    clearSceneGraph();
}

void TestLocalMinDistance::checkAttributes()
{
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

    clearSceneGraph();
}


void TestLocalMinDistance::checkDoubleInit()
{
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

    clearSceneGraph();
}


void TestLocalMinDistance::checkInitReinit()
{
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

    clearSceneGraph();
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
