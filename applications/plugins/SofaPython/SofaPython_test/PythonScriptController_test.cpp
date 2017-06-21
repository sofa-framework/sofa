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

///////////////////////////////////// TESTS ////////////////////////////////////////////////////////
struct PythonScriptController_test : public Sofa_test<>
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

    void checkErrorMessage(bool inPython=true)
    {
        std::stringstream scene ;
        std::string pythonControllerPath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonScriptController.py");
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
           if(inPython)
               pyctrl->draw(nullptr) ;
           else
               pyctrl->onBeginAnimationStep(0.0) ;
        }
    }
};

TEST_F(PythonScriptController_test, checkInvalidCreation)
{
    checkInvalidCreation();
}

TEST_F(PythonScriptController_test, checkExceptionToErrorMessageFromPythonException)
{
    checkErrorMessage(true);
}

TEST_F(PythonScriptController_test, checkExceptionToErrorMessageFromCPPBinding)
{
    checkErrorMessage(false);
}
