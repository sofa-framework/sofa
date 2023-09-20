/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/helper/BackTrace.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
using sofa::component::statecontainer::MechanicalObject ;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <sofa/component/engine/analyze/ClusteringEngine.h>
using sofa::component::engine::analyze::ClusteringEngine ;

using sofa::type::vector;

namespace sofa
{

template <typename _DataTypes>
struct ClusteringEngine_test : public BaseSimulationTest,
        ClusteringEngine<_DataTypes>
{
    typedef ClusteringEngine<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;

    Simulation* m_simu;
    Node::SPtr m_node;
    typename ThisClass::SPtr m_thisObject;
    typename MechanicalObject<DataTypes>::SPtr m_mecaobject;


    void SetUp() override
    {
        m_simu = sofa::simulation::getSimulation();
        ASSERT_NE(m_simu, nullptr);

        m_node = m_simu->createNewGraph("root");
        m_thisObject = New<ThisClass>() ;
        m_mecaobject = New<MechanicalObject<DataTypes>>() ;
        m_mecaobject->init() ;

        m_node->addObject(m_mecaobject) ;
        m_node->addObject(m_thisObject) ;
    }


    // Basic tests (data and init).
    void normalTests(){

        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        EXPECT_TRUE( m_thisObject->findData("useTopo") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("radius") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("number") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("fixedRadius") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("fixedPosition") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("position") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("cluster") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("inFile") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("outFile") != nullptr ) ;

        EXPECT_NO_THROW( m_thisObject->init() ) ;
        EXPECT_NO_THROW( m_thisObject->reinit() ) ;
        EXPECT_NO_THROW( m_thisObject->reset() ) ;

        return ;
    }


    // The draw() function shouldn't crash if no mechanical context
    void drawTest(){

        this->init();

        VisualParams* vparams = sofa::core::visual::visualparams::defaultInstance();
        vparams->displayFlags().setShowBehaviorModels(true);

        EXPECT_NO_THROW(this->draw(vparams));
    }


    // Can not test computation on a simple example because of the overlapping feature
    // Should we run the computation and use the result as a test?
    void updateTest(){

        m_thisObject->init();
        m_thisObject->d_radius = 0.5;
        m_thisObject->d_nbClusters = 3;
        m_thisObject->findData("position")->read("0. 0. 0.    0. 7. 0.   0. 20. 0.    0. 0.1 0.   0. 0.25 0.    0. 7.2 0.    0. 20.2 0.   0. 6.9 0.   0. 0.15 0.");
        m_thisObject->update();

        /*EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[0]));
        EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[1]));
        EXPECT_TRUE(isClusterValid(thisObject->d_cluster.getValue()[2]));*/
    }

    bool isClusterValid(const vector<unsigned int>& cluster)
    {
        const std::vector<unsigned int> result1 = {0,3,4,8};
        const std::vector<unsigned int> result2 = {1,5,7};
        const std::vector<unsigned int> result3 = {2,6};

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

using ::testing::Types;
typedef Types<sofa::defaulttype::Vec3Types> DataTypes;

TYPED_TEST_SUITE(ClusteringEngine_test, DataTypes);

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
