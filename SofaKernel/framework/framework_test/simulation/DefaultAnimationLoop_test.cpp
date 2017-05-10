#include <SofaTest/Sofa_test.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::Message ;
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::MessageAsTestFailure;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::ExecParams;

#include <sofa/simulation/Simulation.h>

namespace sofa
{

struct DefaultAnimationLoop_test : public Sofa_test<>
{

    void testOneStep()
    {
        MessageAsTestFailure raii(Message::Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop />            \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;
        simulation::getSimulation()->animate ( root.get(), (SReal)0.01 );
    }
/*
    void testNegativeStep()
    {
        MessageAsTestFailure raii(Message::Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop />            \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;
        simulation::getSimulation()->animate ( root.get(), (SReal)-0.01 );
    }
*/
};

TEST_F(DefaultAnimationLoop_test, testOneStep ) { testOneStep(); }
//TEST_F(DefaultAnimationLoop_test, testNegativeStep ) { testNegativeStep(); }

}
