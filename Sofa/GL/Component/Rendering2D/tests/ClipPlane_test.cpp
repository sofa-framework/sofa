#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::execparams::defaultInstance; 

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

#include <SofaSimulationGraph/SimpleApi.h>

#include <SofaBaseMechanics/initSofaBaseMechanics.h>

namespace cliplane_test
{

int initMessage(){
    sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
    sofa::simpleapi::importPlugin("Sofa.GL.Component.Rendering3D");

    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestClipPlane : public BaseTest {
public:
    void SetUp() override
    {
        assert(sofa::simulation::getSimulation());
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

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {"position", "normal", "id", "active"};

    for(auto& attrname : attrnames)
        EXPECT_NE( clp->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

    sofa::simulation::getSimulation()->unload(root);
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

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getTreeNode("Level 1")->getObject("clipplane") ;
    ASSERT_NE(clp, nullptr) ;

    sofa::simulation::getSimulation()->unload(root);
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
