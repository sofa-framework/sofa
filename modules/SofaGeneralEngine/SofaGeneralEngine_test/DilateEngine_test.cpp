/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>


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

    Simulation* m_simu;
    Node::SPtr m_node;
    typename ThisClass::SPtr m_thisObject;

    void SetUp()
    {
        setSimulation(m_simu = new DAGSimulation());
        m_node = m_simu->createNewGraph("root");
        m_thisObject = New<ThisClass >() ;
        m_node->addObject(m_thisObject) ;
    }

    // Basic tests (data and init).
    void normalTests(){
        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        EXPECT_TRUE( m_thisObject->findData("input_position") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("output_position") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("triangles") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("quads") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("normal") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("thickness") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("distance") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("minThickness") != NULL ) ;

        EXPECT_NO_THROW( m_thisObject->init() ) ;
        EXPECT_NO_THROW( m_thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( m_thisObject->reinit() ) ;
        EXPECT_NO_THROW( m_thisObject->reset() ) ;
        EXPECT_NO_THROW(this->update()) ;

        return ;
    }

    // Test computation on a simple example
    void updateTest(){
        m_thisObject->findData("position")->read("0. 0. 0.  1. 0. 0.  0. 1. 0.");
        m_thisObject->findData("triangles")->read("0 1 2");
        m_thisObject->findData("distance")->read("0.");
        m_thisObject->init();
        m_thisObject->update();

        // Check output
        EXPECT_TRUE(m_thisObject->findData("output_position")->getValueString()=="0 0 0 1 0 0 0 1 0"); // Should stay invariant
        EXPECT_TRUE(m_thisObject->findData("normal")->getValueString()=="0 0 1 0 0 1 0 0 1");

        m_thisObject->findData("distance")->read("0.1");
        m_thisObject->update();
        EXPECT_TRUE(m_thisObject->findData("output_position")->getValueString()=="0 0 0.1 1 0 0.1 0 1 0.1"); // Should apply distance along normal
    }

};

using testing::Types;
#ifdef SOFA_WITH_DOUBLE
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(DilateEngine_test, DataTypes);
TYPED_TEST(DilateEngine_test, NormalBehavior) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(DilateEngine_test, UpdateTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->updateTest()) ;
}
#endif

}
