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

int initMessage(){
    //MessageDispatcher::clearHandlers() ;
    //MessageDispatcher::addHandler(new ClangMessageHandler()) ;
    return 0;
}
int messageInited = initMessage();

namespace sofa {

struct TestLightManager : public Sofa_test<double> {
};

void checkAttributes()
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


TEST_F(TestLightManager, checkAttributes)
{
    checkAttributes();
}

}
