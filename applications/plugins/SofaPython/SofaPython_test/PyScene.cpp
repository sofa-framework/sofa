#include <SofaTest/Sofa_test.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <SofaPython/PythonScriptController.h>

namespace sofa {

struct PyScene_test : public Sofa_test<>
{
    PyScene_test()
    {
    }

    bool loadPySceneWithPyCtrlr()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PySceneWithPyCtlr.py");
        //std::cout<<scenePath<<std::endl;

        //Init

        simulation::Simulation* simulation;
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Scene creation
        simulation::Node::SPtr root =  simulation::getSimulation()->load ( scenePath.c_str() );
        simulation::getSimulation()->init(root.get());

        // get the python script controller
        component::controller::PythonScriptController *ctr = root->get<component::controller::PythonScriptController>();
        if (!ctr)
            return false; // no script controller ??

        if (!ctr->scriptControllerInstance())
            return false; // script not loaded correctly

        //std::cout<<ctr->getName()<<std::endl;

        return true;
    }

    bool loadXmlSceneWithPyCtrlr()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_XmlSceneWithPyCtlr.scn");
        //std::cout<<scenePath<<std::endl;

        //Init

        simulation::Simulation* simulation;
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Scene creation
        simulation::Node::SPtr root =  simulation::getSimulation()->load ( scenePath.c_str() );
        simulation::getSimulation()->init(root.get());

        // get the python script controller
        component::controller::PythonScriptController *ctr = root->get<component::controller::PythonScriptController>();
        if (!ctr)
            return false; // no script controller ??

        if (!ctr->scriptControllerInstance())
            return false; // script not loaded correctly

        //std::cout<<ctr->getName()<<std::endl;

        return true;
    }

};

TEST_F( PyScene_test, XmlScene)
{
    EXPECT_TRUE( this->loadXmlSceneWithPyCtrlr() );
}

TEST_F( PyScene_test, PythonScene)
{
    EXPECT_TRUE( this->loadPySceneWithPyCtrlr() );
}

}// namespace sofa
