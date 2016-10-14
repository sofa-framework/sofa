#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>
#include <SofaBaseMechanics/MechanicalObject.h>
using sofa::component::container::MechanicalObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/AverageCoord.h>
using sofa::component::engine::AverageCoord ;

using sofa::helper::vector;


namespace sofa
{

template <typename _DataTypes>
struct AverageCoord_test : public Sofa_test<typename _DataTypes::Real>,
        AverageCoord<_DataTypes>
{
    typedef AverageCoord<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;


    // Basic tests (data and init).
    void normalTests(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename MechanicalObject<DataTypes>::SPtr mecaobject = New<MechanicalObject<DataTypes> >() ;
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        node->addObject(mecaobject) ;
        mecaobject->init();

        node->addObject(thisObject) ;

        thisObject->setName("myname") ;
        EXPECT_TRUE(thisObject->getName() == "myname") ;

        EXPECT_TRUE( thisObject->findData("indices") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("vecId") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("average") != NULL ) ;

        EXPECT_NO_THROW( thisObject->init() ) ;
        EXPECT_NO_THROW( thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( thisObject->reinit() ) ;
        EXPECT_NO_THROW( thisObject->reset() ) ;

        this->mstate = NULL;
        EXPECT_NO_THROW(this->update()) ;

        return ;
    }


    // Test computation on a simple example
    void updateTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename MechanicalObject<DataTypes>::SPtr mecaobject = New<MechanicalObject<DataTypes> >() ;
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        node->addObject(mecaobject) ;
        mecaobject->findData("position")->read("0. 0. 0.   1. 0. 0.   2. 4. 0.   3. 0. 0.");
        mecaobject->init();

        node->addObject(thisObject) ;
        thisObject->findData("indices")->read("0 1 2 3");
        thisObject->init();
        thisObject->update();

        EXPECT_TRUE(thisObject->findData("average")->getValueString()=="1.5 1 0");
    }

};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(AverageCoord_test, DataTypes);

TYPED_TEST(AverageCoord_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(AverageCoord_test, UpdateTest) {
    ASSERT_NO_THROW(this->updateTest()) ;
}


}
