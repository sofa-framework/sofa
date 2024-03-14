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

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <sofa/component/engine/transform/DifferenceEngine.h>
using sofa::component::engine::transform::DifferenceEngine ;

using sofa::type::vector;

namespace sofa
{

template <typename _DataTypes>
struct DifferenceEngine_test : public BaseSimulationTest,
        DifferenceEngine<_DataTypes>
{
    typedef DifferenceEngine<_DataTypes> ThisClass ;
    typedef _DataTypes DataTypes;

    Simulation* m_simu;
    Node::SPtr m_node;
    typename ThisClass::SPtr m_thisObject;

    void SetUp() override
    {
        m_simu = sofa::simulation::getSimulation();
        ASSERT_NE(m_simu, nullptr);

        m_node = m_simu->createNewGraph("root");
        m_thisObject = New<ThisClass >() ;
        m_node->addObject(m_thisObject) ;
    }

    // Basic tests (data and init).
    void normalTests(){

        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        EXPECT_TRUE( m_thisObject->findData("input") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("substractor") != nullptr ) ;
        EXPECT_TRUE( m_thisObject->findData("output") != nullptr ) ;

        EXPECT_NO_THROW( m_thisObject->init() ) ;
        EXPECT_NO_THROW( m_thisObject->reinit() ) ;
        EXPECT_NO_THROW( m_thisObject->reset() ) ;

        return ;
    }


    // Test computation on a simple example
    void updateTest(){

        m_thisObject->findData("input")->read("0. 0.5 0.5  0. 0. 1.  0. -1. 3.");
        m_thisObject->findData("substractor")->read("0. 0. 0.5  0. 1. 1.  0. 1. 2.");
        m_thisObject->update();

        EXPECT_TRUE(m_thisObject->findData("output")->getValueString() == "0 0.5 0 0 -1 0 0 -2 1");
    }


    // Shouldn't crash if input and substractor have different size
    void dataTest(){

        m_thisObject->findData("input")->read("0. 0. 0.");
        m_thisObject->findData("substractor")->read("0. 0. 0. 0. 0. 0.");
        EXPECT_NO_THROW(m_thisObject->update());
    }

};

using ::testing::Types;
typedef Types<type::Vec3> DataTypes;

TYPED_TEST_SUITE(DifferenceEngine_test, DataTypes);

TYPED_TEST(DifferenceEngine_test, NormalBehavior) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(DifferenceEngine_test, UpdateTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->updateTest()) ;
}

TYPED_TEST(DifferenceEngine_test, DataTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->dataTest()) ;
}

}
