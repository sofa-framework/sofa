#include <SofaTest/Sofa_test.h>

#include <SofaPython/PythonFactory.h>
#include <SofaPython/Binding_BaseObject.h>


#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/core/ObjectFactory.h>

#include "../SceneLoaderPY.h"
using sofa::simulation::SceneLoaderPY ;
using sofa::core::objectmodel::BaseObject ;

namespace sofa {

/////////////////////////////// A new Component ////////////////////////////////////////////////////
class ExternalComponent : public core::objectmodel::BaseObject
{
public:
    Data<std::string> d_value ;

    SOFA_CLASS( ExternalComponent,core::objectmodel::BaseObject);

    ExternalComponent() :
        d_value(initData(&d_value, "value", "value", "simpletest"))
    {
    }

    void helloWorld()
    {
        msg_info()<<"ExternalComponent \""<<this->getName()<<"\" says hello world";
        ++nbcalls;
    }

    static int nbcalls;
};

int ExternalComponent::nbcalls = 0;



//////////////////// //////// Registering the new component in the factory /////////////////////////
SOFA_DECL_CLASS (ExternalComponent)
int ExternalComponentClass = core::RegisterObject ( "An dummy External Component" )
        .add<ExternalComponent>(true);
}



////////////////////////////// Binding the new component in Python /////////////////////////////////
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
//////////////////////// End of the new component  /////////////////////////////////////////////////



///////////////////////////////////// TESTS ////////////////////////////////////////////////////////
namespace sofa {
struct PythonFactory_test : public Sofa_test<>,
                            public ::testing::WithParamInterface<std::vector<std::string>>
{
protected:

    virtual void SetUp() override
    {
        /// ADDING new component in the python Factory
        /// of course its binding must be defined!
        SP_ADD_CLASS_IN_FACTORY( ExternalComponent, sofa::ExternalComponent )
    }

    virtual void TearDown() override
    {
    }

    void test()
    {
        EXPECT_EQ(0, ExternalComponent::nbcalls);
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonFactory.py");
        simulation::getSimulation()->load(scenePath.c_str());

        EXPECT_EQ(1, ExternalComponent::nbcalls);
    }


    void testAttributeConversion(const std::vector<std::string>& params)
    {
        std::string value = params[0] ;
        std::string result = params[1] ;

        const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonFactory2.py");
        std::ofstream f(scenePath, std::ofstream::out);
        ASSERT_TRUE(f.is_open());
        f <<
                 "import Sofa                          \n"
                 "from SofaTest.Macro import *         \n"
                 "                                     \n"
                 "class NonCustomizedObject(object):   \n"
                 "   def __init__(self):               \n"
                 "        return None                  \n"
                 "   def __str__(self):                \n"
                 "        return 'default'             \n"
                 "class CustomObject(object):           \n"
                 "   def getAsACreateObjectParameter(self):            \n"
                 "        return 'custom value'        \n"
                 "def createScene(rootNode):           \n"
                 "    first = rootNode.createObject( 'ExternalComponent', name='theFirst') \n"
                 "    externalComponent = rootNode.createObject( 'ExternalComponent', name='second', value="<<value<<") \n"
                 "    externalComponent.helloWorld()   \n" ;
        f.close();

        Node::SPtr root = simulation::getSimulation()->load(scenePath.c_str());
        ASSERT_NE(root, nullptr);

        BaseObject* aComponent = root->getObject("second") ;

        ASSERT_NE(aComponent, nullptr);
        ASSERT_NE(aComponent->findData("value"), nullptr);
        ASSERT_EQ(aComponent->findData("value")->getValueString(), result);
        simulation::getSimulation()->unload(root);
    }
};

TEST_F(PythonFactory_test, result)
{
    test();
}


TEST_F(PythonFactory_test, testCreateObjectDataConversionWarning)
{
    EXPECT_MSG_EMIT(Warning) ;
    this->testAttributeConversion({"NonCustomizedObject()", "default"});
}

std::vector<std::vector<std::string>> dataconversionvalues =
    {{"1", "1"},
     {"1.1", "1.1"},
     {"True", "True"},
     {"'aString'", "aString"},
     {"'aString'.join('[ ]')", "[aString aString]"},
     {"' '.join(['AA', 'BB', 'CC'])", "AA BB CC"},
     {"[1, 2, 3, 4]", "1 2 3 4"},
     {"[1.0, 2.0, 3.0, 4.0]", "1.0 2.0 3.0 4.0"},
     {"['ab', 'cd', 'ef', 'gh']", "ab cd ef gh"},
     {"[[1,2], [3,4], [5,6]]", "1 2 3 4 5 6"},
     {"[['aa','bb'], ['cc','dd'], ['ee','ff']]", "aa bb cc dd ee ff"},
     {"range(1,5)", "1 2 3 4"},
     {"xrange(1,5)", "1 2 3 4"},
     {"'XX_'+first.findData('name').getLinkPath()", "XX_@/theFirst.name"},
     {"first.findData('name').getLinkPath()", "theFirst"},
     {"first.findData('name')", "theFirst"},
     {"'XX_'+rootNode.getAsACreateObjectParameter()", "XX_@"},
     {"CustomObject()", "custom value"}
    } ;

TEST_P(PythonFactory_test, testCreateObjectDataConversion)
{
   this->testAttributeConversion(GetParam());
}

INSTANTIATE_TEST_CASE_P(testCreateObjectDataConversion,
                        PythonFactory_test,
                        ::testing::ValuesIn(dataconversionvalues));
}


