#include <regex>
#include <vector>
#include <fstream>

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

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

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
struct PythonScriptController_test : public Sofa_test<>,
        public ::testing::WithParamInterface<std::string>
{
protected:
    virtual void SetUp() override
    {
    }

    virtual void TearDown() override
    {
    }

    void checkInvalidCreation()
    {
        EXPECT_MSG_EMIT(Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >                   \n"
                 "      <RequiredPlugin name='SofaPython' />                                     \n"
                 "      <PythonScriptController classname='AClass' />                            \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;
    }

    void checkErrorMessage(const std::string& teststring)
    {
        std::string pythonControllerPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonScriptController_AutoGen.py");

        std::ofstream f(pythonControllerPath);
        std::string pytmp = R"(
import Sofa
def f3():
    raise ValueError('The value is not valid')

class TestController(Sofa.PythonScriptController):
    def __init__(self):
        return None

    def anInvalidFunction(self):
        name = self.findData("name")
        name.setValue(1)

    def f2(self):
        f3()

    def draw(self):
        $line
)";
        //TODO(dmarchal): I do not use regex_replace because clang 3.4 we use in our CI is buggy.
        //After 2018 please restore back the regex_replace version
        //pytmp = std::regex_replace(pytmp, std::regex("\\$line"), teststring);
        ReplaceSubstring(pytmp, std::string("$line"), teststring) ;

        f << pytmp ;
        f.close();

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
              <<   "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >                           \n"
              <<   "      <RequiredPlugin name='SofaPython' />                                             \n"
              <<   "      <PythonScriptController classname='TestController' filename='"<<pythonControllerPath<< "'/>    \n"
              <<   "</Node>                                                                                \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        PythonScriptController* pyctrl = root->getTreeObject<PythonScriptController>();
        ASSERT_NE(pyctrl, nullptr) ;

        /// This function rise an exception
        /// The exception should propage up to the Sofa Layer.
        {
           EXPECT_MSG_EMIT(Error) ;
           pyctrl->draw(nullptr) ;
        }
    }
};

TEST_F(PythonScriptController_test, checkInvalidCreation)
{
    checkInvalidCreation();
}


TEST_F(PythonScriptController_test, checkExceptionToErrorMessageFromPythonException)
{
    checkErrorMessage("self.f2()");
}


TEST_P(PythonScriptController_test, checkErrorMessageFromCPPBinding)
{
    checkErrorMessage(GetParam());
}

std::vector<std::string> testvalues = {
    "self.anInvalidFunction()",
    "self.name = 5",
    "Sofa.BaseContext.getObject(1234, 'WillNotWork')",
    "Sofa.Topology.setNbPoints(1234)",
    "Sofa.BaseContext.getObject(self.findData('name'), 'WillNotWork')",
    "Sofa.BaseContext.getObject(None, 'WillNotWork')"
};

INSTANTIATE_TEST_CASE_P(checkErrorMesageFromCPPBinding,
                        PythonScriptController_test,
                        ::testing::ValuesIn(testvalues));


