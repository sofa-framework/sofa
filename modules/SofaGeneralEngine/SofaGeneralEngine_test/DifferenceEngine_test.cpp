#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaGeneralEngine/DifferenceEngine.h>
using sofa::component::engine::DifferenceEngine ;

using sofa::helper::vector;

namespace sofa
{

template <typename _DataTypes>
struct DifferenceEngine_test : public Sofa_test<typename _DataTypes::value_type>,
        DifferenceEngine<_DataTypes>
{
    typedef DifferenceEngine<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;


    // Basic tests (data and init).
    void normalTests(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        thisObject->setName("myname") ;
        EXPECT_TRUE(thisObject->getName() == "myname") ;

        EXPECT_TRUE( thisObject->findData("input") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("substractor") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("output") != NULL ) ;

        EXPECT_NO_THROW( thisObject->init() ) ;
        EXPECT_NO_THROW( thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( thisObject->reinit() ) ;
        EXPECT_NO_THROW( thisObject->reset() ) ;

        return ;
    }


    // Test computation on a simple example
    void updateTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        thisObject->findData("input")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        thisObject->findData("substractor")->read("0. 0. 0.5  0. 1. 1.  0. 1. 2.");
        thisObject->update();

        EXPECT_TRUE(thisObject->findData("output")->getValueString() == "0 0.5 0 0 -1 0 0 -2 1");
    }


    // Shouldn't crash if input and substractor have different size
    void dataTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        thisObject->findData("input")->read("0. 0. 0.");
        thisObject->findData("substractor")->read("0. 0. 0. 0. 0. 0.");
        EXPECT_NO_THROW(thisObject->update());
    }

};

using testing::Types;
typedef Types<defaulttype::Vec3d> DataTypes;

TYPED_TEST_CASE(DifferenceEngine_test, DataTypes);

TYPED_TEST(DifferenceEngine_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(DifferenceEngine_test, UpdateTest) {
    ASSERT_NO_THROW(this->updateTest()) ;
}

TYPED_TEST(DifferenceEngine_test, DataTest) {
    ASSERT_NO_THROW(this->dataTest()) ;
}

}
