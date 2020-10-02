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

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

namespace cliplane_test
{

int initMessage(){
    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestClipPlane : public Sofa_test<> {
public:
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
