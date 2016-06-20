#include <gtest/gtest.h>

#include <SofaPython/PythonFactory.h>
#include <SofaPython/Binding_BaseObject.h>


#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/ObjectFactory.h>



namespace sofa {




////// A new Component ////////

class ExternalComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( ExternalComponent,core::objectmodel::BaseObject);

    void helloWorld()
    {
        std::cerr<<"ExternalComponent \""<<this->getName()<<"\" says hello world"<<std::endl;
        ++nbcalls;
    }

    static int nbcalls;

};

int ExternalComponent::nbcalls = 0;


//////// Registering the new component in the factory //////////


SOFA_DECL_CLASS (ExternalComponent)

int ExternalComponentClass = core::RegisterObject ( "An dummy External Component" )
        .add<ExternalComponent>(true);


}



//////// Binding the new component in Python //////////


SP_DECLARE_CLASS_TYPE(ExternalComponent)

extern "C" PyObject * ExternalComponent_helloWorld(PyObject *self, PyObject * /*args*/)
{
    sofa::ExternalComponent* obj= down_cast<sofa::ExternalComponent>(((PySPtr<sofa::core::objectmodel::Base>*)self)->object->toBaseObject());
    obj->helloWorld();
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(ExternalComponent)
SP_CLASS_METHOD(ExternalComponent,helloWorld)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(ExternalComponent,sofa::ExternalComponent,BaseObject)




//////// End of the new component  //////////






namespace sofa {


struct PythonFactory_test : public ::testing::Test
{
protected:

    virtual void SetUp()
    {
        // ADDING new component in the python Factory
        // of course its binding must be defined!
        SP_ADD_CLASS_IN_FACTORY( ExternalComponent, sofa::ExternalComponent )

    }

    void test()
    {
        EXPECT_EQ(0, ExternalComponent::nbcalls);

        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonFactory.py");
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
        simulation::getSimulation()->load(scenePath.c_str());

        EXPECT_EQ(1, ExternalComponent::nbcalls);
    }

};

TEST_F(PythonFactory_test, result)
{
    test();
}

}
