#include <vector>
using std::vector;

#include <string>
using std::string;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

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

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::MainCountingMessageHandler;

#include <sofa/helper/logging/LoggingMessageHandler.h>
using sofa::helper::logging::MainLoggingMessageHandler;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

namespace cliplane_test
{

int initMessage(){
    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    MessageDispatcher::addHandler(&MainLoggingMessageHandler::getInstance()) ;
    MessageDispatcher::addHandler(&MainCountingMessageHandler::getInstance()) ;
    return 0;
}

int messageInited = initMessage();

class TestClipPlane : public Sofa_test<double> {
public:
    void checkClipPlaneValidAttributes();
    void checkClipPlaneAttributesValues(const std::string& dataname, const std::string& value);
};

void TestClipPlane::checkClipPlaneValidAttributes()
{
    MessageAsTestFailure warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject/>                                                         \n"
             "   <ClipPlane name='clipplane'/>                                               \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {"position", "normal", "id", "active"};

    for(auto& attrname : attrnames)
        EXPECT_NE( clp->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    clearSceneGraph();
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

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

TEST_F(TestClipPlane, checkClipPlaneIdInValidValues_OpenIssue)
{
    ExpectMessage warning(Message::Warning) ;

    checkClipPlaneAttributesValues("id", "-1");
}

TEST_F(TestClipPlane, checkClipPlaneIdTooBigValue_OpenIssue)
{
    ExpectMessage warning(Message::Warning) ;

    checkClipPlaneAttributesValues("id", "15654654");
}

TEST_F(TestClipPlane, checkClipPlaneNormalInvalidNormalValue)
{
    ExpectMessage warning(Message::Warning) ;

    checkClipPlaneAttributesValues("normal", "1 0");
}

// Normal should be "normalized" so passing a non normalized value should be tested and report an error
// and convert it to a noramlized version.
TEST_F(TestClipPlane, checkClipPlanePassingNotNormalizedAsNormal_OpenIssue)
{
    ExpectMessage warning(Message::Warning) ;

    checkClipPlaneAttributesValues("normal", "2 0 0");
}


} // cliplane_test
