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

#include <SofaGeneralEngine/ComplementaryROI.h>
using sofa::component::engine::ComplementaryROI ;

using sofa::helper::vector;

namespace sofa
{

template <typename _DataTypes>
struct ComplementaryROI_test : public Sofa_test<typename _DataTypes::Real>,
        ComplementaryROI<_DataTypes>
{
    typedef ComplementaryROI<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;


    // Basic tests (data and init).
    void normalTests(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        thisObject->setName("myname") ;
        EXPECT_TRUE(thisObject->getName() == "myname") ;

        EXPECT_TRUE( thisObject->findData("position") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("nbSet") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("indices") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("pointsInROI") != NULL ) ;

        EXPECT_NO_THROW( thisObject->init() ) ;
        EXPECT_NO_THROW( thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( thisObject->reinit() ) ;
        EXPECT_NO_THROW( thisObject->reset() ) ;

        thisObject->findData("nbSet")->read("3");
        thisObject->init();

        EXPECT_TRUE( thisObject->findData("setIndices1") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("setIndices2") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("setIndices3") != NULL ) ;

        return ;
    }


    // Test computation on a simple example
    void updateTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        thisObject->findData("position")->read("0. 0. 0.  0. 0. 1.  0. 0. 2.  0. 0. 3.  0. 0. 4.  0. 0. 5.  0. 0. 6.");
        thisObject->findData("nbSet")->read("2");
        thisObject->init();
        thisObject->findData("setIndices1")->read("0 1");
        thisObject->findData("setIndices2")->read("5 6");
        thisObject->update();

        EXPECT_TRUE(thisObject->findData("indices")->getValueString() == "2 3 4");
        EXPECT_TRUE(thisObject->findData("pointsInROI")->getValueString() == "0 0 2 0 0 3 0 0 4");
    }


};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(ComplementaryROI_test, DataTypes);

TYPED_TEST(ComplementaryROI_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(ComplementaryROI_test, UpdateTest) {
    ASSERT_NO_THROW(this->updateTest()) ;
}

}
