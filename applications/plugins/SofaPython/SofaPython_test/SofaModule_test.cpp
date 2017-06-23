#include <regex>
#include <vector>

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaPython/PythonFactory.h>
#include <SofaPython/Binding_BaseObject.h>


#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/ObjectFactory.h>
#include <SofaPython/PythonToSofa.inl>

using sofa::simulation::Node;

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
};

TEST_F(SofaModule_test, getAvailableComponents)
{
    checkGetComponentList();
}

