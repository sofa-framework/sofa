#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa ;
using namespace sofa::simpleapi ;
using namespace sofa::simpleapi::components ;

class SimpleApi_test : public BaseSimulationTest
{
public:
    bool testParamAPI();
    bool testParamString();
};

bool SimpleApi_test::testParamAPI()
{
    Simulation::SPtr simu = createSimulation("DAG") ;
    Node::SPtr root = createRootNode(simu, "root") ;

    auto meca1 = createObject(root, MechanicalObject::objectname, {
                     {MechanicalObject::data::name, "aMechanicalObject1"},
                     {MechanicalObject::data::position, "1 2 3"}
                 });


    auto meca2 = createObject(root, MechanicalObject::objectname, {
                     {MechanicalObject::data::name, "aMechanicalObject2"},
                     {MechanicalObject::data::position, "1 2 3"}
                 });

    EXPECT_EQ( (meca1->getName()), std::string("aMechanicalObject1") ) ;
    EXPECT_EQ( (meca2->getName()), std::string("aMechanicalObject2") ) ;

    return true ;
}

bool SimpleApi_test::testParamString()
{
    Simulation::SPtr simu = createSimulation("DAG") ;
    Node::SPtr root = createRootNode(simu, "root") ;

    auto meca1 = createObject(root, "MechanicalObject", {
                     {"name", "aMechanicalObject1"},
                     {"position", "1 2 3"}
                 });

    auto meca2 = createObject(root, "MechanicalObject", {
                     {"name", "aMechanicalObject2"},
                     {"position", "1 2 3"}
                 });

    EXPECT_EQ( (meca1->getName()), std::string("aMechanicalObject1") ) ;
    EXPECT_EQ( (meca2->getName()), std::string("aMechanicalObject2") ) ;

    return true;
}


TEST_F(SimpleApi_test, testParamAPI )
{
    ASSERT_TRUE( testParamAPI() );
}

TEST_F(SimpleApi_test, createParamString )
{
    ASSERT_TRUE( testParamString() );
}
