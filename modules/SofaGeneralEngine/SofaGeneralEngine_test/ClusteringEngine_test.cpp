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

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaGeneralEngine/ClusteringEngine.h>
using sofa::component::engine::ClusteringEngine ;

using sofa::helper::vector;

namespace sofa
{

template <typename _DataTypes>
struct ClusteringEngine_test : public Sofa_test<typename _DataTypes::Real>,
        ClusteringEngine<_DataTypes>
{
    typedef ClusteringEngine<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;


    // Basic tests (data and init).
    void normalTests(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename MechanicalObject<DataTypes>::SPtr mecaobject = New<MechanicalObject<DataTypes> >() ;
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;
        mecaobject->init() ;

        node->addObject(mecaobject) ;
        node->addObject(thisObject) ;

        thisObject->setName("myname") ;
        EXPECT_TRUE(thisObject->getName() == "myname") ;

        EXPECT_TRUE( thisObject->findData("useTopo") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("radius") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("number") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("fixedRadius") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("fixedPosition") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("position") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("cluster") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("inFile") != NULL ) ;
        EXPECT_TRUE( thisObject->findData("outFile") != NULL ) ;

        EXPECT_NO_THROW( thisObject->init() ) ;
        EXPECT_NO_THROW( thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( thisObject->reinit() ) ;
        EXPECT_NO_THROW( thisObject->reset() ) ;

        return ;
    }


    // The draw() function shouldn't crash if no mechanical context
    void drawTest(){

        typename ThisClass::SPtr thisObject = New<ThisClass>() ;
        thisObject->init();

        VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
        vparams->displayFlags().setShowBehaviorModels(true);

        EXPECT_NO_THROW(thisObject->draw(vparams));
    }


    // Can not test computation on a simple example because of the overlapping feature
    // Should we run the computation and use the result as a test?
    void updateTest(){
        Simulation* simu;
        setSimulation(simu = new DAGSimulation());

        Node::SPtr node = simu->createNewGraph("root");
        typename MechanicalObject<DataTypes>::SPtr mecaobject = New<MechanicalObject<DataTypes> >() ;
        typename ThisClass::SPtr thisObject = New<ThisClass >() ;

        node->addObject(mecaobject);
        mecaobject->init();

        node->addObject(thisObject) ;
        thisObject->init();
        thisObject->d_radius = 0.5;
        thisObject->d_nbClusters = 3;
        thisObject->findData("position")->read("0. 0. 0.    0. 7. 0.   0. 20. 0.    0. 0.1 0.   0. 0.25 0.    0. 7.2 0.    0. 20.2 0.   0. 6.9 0.   0. 0.15 0.");
        thisObject->update();

        /*EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[0]));
        EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[1]));
        EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[2]));*/
    }

    bool isClusterValid(const vector<unsigned int>& cluster)
    {
        std::vector<unsigned int> result1 = {0,3,4,8};
        std::vector<unsigned int> result2 = {1,5,7};
        std::vector<unsigned int> result3 = {2,6};

        vector<unsigned int> sortedCluster(cluster);
        sort(sortedCluster.begin(),sortedCluster.end());

        if(sortedCluster == result1)
            return true;
        if(sortedCluster == result2)
            return true;
        if(sortedCluster == result3)
            return true;

        return false;
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(ClusteringEngine_test, DataTypes);

TYPED_TEST(ClusteringEngine_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(ClusteringEngine_test, DrawTest) {
    ASSERT_NO_THROW(this->drawTest()) ;
}

TYPED_TEST(ClusteringEngine_test, UpdateTest) {
    ASSERT_NO_THROW(this->updateTest()) ;
}

}
