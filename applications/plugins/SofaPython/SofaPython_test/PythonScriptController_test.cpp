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

};

TEST_F(PythonScriptController_test, checkInvalidCreation)
{
    checkInvalidCreation();
}

