#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/DilateEngine.h>
using sofa::component::engine::DilateEngine ;

using sofa::helper::vector;


namespace sofa
{

template <typename _DataTypes>
struct DilateEngine_test : public Sofa_test<typename _DataTypes::Real>,
        DilateEngine<_DataTypes>
{
    typedef DilateEngine<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;


    // Basic tests (data and init).
    void normalTests(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;
        node->addObject(thisObject) ;

        thisObject->setName("myname") ;
        EXPECT_TRUE(thisObject->getName() == "myname") ;

        EXPECT_TRUE( thisObject->findData("input_position") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("output_position") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("triangles") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("quads") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("normal") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("thickness") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("distance") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("minThickness") != NULL ) ;

        EXPECT_NO_THROW( thisObject->init() ) ;
        EXPECT_NO_THROW( thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( thisObject->reinit() ) ;
        EXPECT_NO_THROW( thisObject->reset() ) ;
        EXPECT_NO_THROW(this->update()) ;

        return ;
    }

    // Test computation on a simple example
    void updateTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        node->addObject(thisObject) ;
        thisObject->findData("position")->read("0. 0. 0.  1. 0. 0.  0. 1. 0.");
        thisObject->findData("triangles")->read("0 1 2");
        thisObject->findData("distance")->read("0.");
        thisObject->init();
        thisObject->update();

        // Check output
        EXPECT_TRUE(thisObject->findData("output_position")->getValueString()=="0 0 0 1 0 0 0 1 0"); // Should stay invariant
        EXPECT_TRUE(thisObject->findData("normal")->getValueString()=="0 0 1 0 0 1 0 0 1");

        thisObject->findData("distance")->read("0.1");
        thisObject->update();
        EXPECT_TRUE(thisObject->findData("output_position")->getValueString()=="0 0 0.1 1 0 0.1 0 1 0.1"); // Should apply distance along normal
    }

};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(DilateEngine_test, DataTypes);

TYPED_TEST(DilateEngine_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(DilateEngine_test, UpdateTest) {
    ASSERT_NO_THROW(this->updateTest()) ;
}

}
