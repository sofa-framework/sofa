#include <regex>
#include <vector>

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaPython/PythonFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaPython/Binding_BaseObject.h>


#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;
using sofa::Data ;

#include <sofa/core/ObjectFactory.h>
#include <SofaPython/PythonToSofa.inl>

#include <sofa/core/ObjectFactory.h>

#include <SofaPython/SceneLoaderPY.h>
using sofa::simulation::SceneLoaderPY ;
using sofa::core::objectmodel::BaseObject ;
using sofa::core::ExecParams ;

#include <SofaPython/PythonScriptController.h>
using sofa::component::controller::PythonScriptController ;

template <typename charType>
void ReplaceSubstring(std::basic_string<charType>& subject,
    const std::basic_string<charType>& search,
    const std::basic_string<charType>& replace)
{
    if (search.empty()) { return; }
    typename std::basic_string<charType>::size_type pos = 0;
    while((pos = subject.find(search, pos)) != std::basic_string<charType>::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

#include <fstream>

///////////////////////////////////// TESTS ////////////////////////////////////////////////////////
struct SofaModule_test : public Sofa_test<>,
        public ::testing::WithParamInterface<std::string>
{
protected:
    virtual void SetUp() override
    {
    }

    virtual void TearDown() override
    {
    }

    void testMsgInfo()
    {
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_SofaModule.py");

        {
            EXPECT_MSG_EMIT(Info, Error) ;
            sofa::simulation::getSimulation()->load(scenePath.c_str());
        }
    }

    void checkGetComponentList()
    {
        std::string pythonControllerPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_AutoGen.py");

        std::ofstream f(pythonControllerPath);
        std::string pytmp = R"(
import Sofa

def createScene(rootNode):
    Sofa.msg_info(str(Sofa.getAvailableComponents()))
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

	void checkTimerSetOutputType(const std::string& timer_name, const std::string& output_type)
	{
		std::string pythonControllerPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_AutoGen.py");

		std::ofstream f(pythonControllerPath);
		std::string pytmp = std::string("import Sofa\n\ndef createScene(rootNode):\n"
										"\tSofa.timerSetOutputType(\"")
										+ timer_name + std::string("\", \"")
										+ output_type + std::string("\")\n");

		f << pytmp ;
		f.close();

		{
			// sofa init
			sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

			// load scene
			Node::SPtr root = sofa::simulation::getSimulation()->load(pythonControllerPath.c_str());

			ASSERT_NE(root, nullptr);

			ASSERT_TRUE(sofa::helper::AdvancedTimer::getOutputType(timer_name) == sofa::helper::AdvancedTimer::convertOutputType(output_type));
			sofa::simulation::getSimulation()->unload(root);
		}
	}
};

TEST_F(SofaModule_test, getAvailableComponents)
{
    checkGetComponentList();
}

TEST_F(SofaModule_test,  testMsgInfo)
{
    this->testMsgInfo();
}

TEST_F(SofaModule_test, timerSetOutPutType)
{
	this->checkTimerSetOutputType("validID", "JSON");
	this->checkTimerSetOutputType("", "JSON");
	this->checkTimerSetOutputType("invalid", "JSON");
	this->checkTimerSetOutputType("validID", "LJSON");
	this->checkTimerSetOutputType("validID", "STDOUT");
	this->checkTimerSetOutputType("validID", "");
	this->checkTimerSetOutputType("validID", "invalidType");
}
