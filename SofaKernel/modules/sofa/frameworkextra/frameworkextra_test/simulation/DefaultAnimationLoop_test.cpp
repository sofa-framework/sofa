#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::ExecParams;

#include <sofa/simulation/Simulation.h>

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

namespace sofa
{

struct DefaultAnimationLoop_test : public BaseSimulationTest
{

    void testOneStep()
    {
        EXPECT_MSG_NOEMIT(Error) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop />            \n"
                 "</Node>                                                                        \n" ;

        SceneInstance c("xml", scene.str()) ;
        Node* root = c.root.get() ;
        ASSERT_NE(root, nullptr) ;

        c.initScene() ;
        sofa::simulation::getSimulation()->animate ( root, (SReal)0.01 );
    }

};

TEST_F(DefaultAnimationLoop_test, testOneStep ) { testOneStep(); }

}
