#include <SofaTest/Sofa_test.h>

#include <SofaPython/PythonFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaPython/Binding_BaseObject.h>

#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;
using sofa::Data ;

#include "../SceneLoaderPY.h"
using sofa::simulation::SceneLoaderPY ;
using sofa::core::objectmodel::BaseObject ;

///////////////////////////////////// TESTS ////////////////////////////////////////////////////////
struct SofaModule_test : public sofa::Sofa_test<>,
                         public ::testing::WithParamInterface<std::vector<std::string>>
{
protected:
    void testMsgInfo()
    {
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_SofaModule.py");

        {
            EXPECT_MSG_EMIT(Info, Error) ;
            sofa::simulation::getSimulation()->load(scenePath.c_str());
        }
    }

	void checkTimerSetOutputType()
	{
		std::string pythonControllerPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_AutoGen.py");

		std::ofstream f(pythonControllerPath);
		std::string pytmp = R"(
import Sofa

def createScene(rootNode):
	Sofa.timerSetOutPutType("validID", "JSON")
	Sofa.timerSetOutPutType("", "JSON")
	Sofa.timerSetOutPutType("invalid", "JSON")
	Sofa.timerSetOutPutType("validID", "LJSON")
	Sofa.timerSetOutPutType("validID", "STDOUT")
	Sofa.timerSetOutPutType("validID", "")
	Sofa.timerSetOutPutType("validID", "invalidType")
)";

		f << pytmp ;
		f.close();

		{
			// sofa init
			sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

			// load scene
			sofa::simulation::getSimulation()->load(pythonControllerPath.c_str());
		}
	}
};

TEST_F(SofaModule_test,  testMsgInfo)
{
    this->testMsgInfo();
}

TEST_F(SofaModule_test, timerSetOutPutType)
{
	this->checkTimerSetOutputType();
}
