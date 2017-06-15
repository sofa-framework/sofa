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

#include <SofaGeneralEngine/ExtrudeEdgesAndGenerateQuads.h>
using sofa::component::engine::ExtrudeEdgesAndGenerateQuads ;

using sofa::helper::vector;


namespace sofa
{

template <typename _DataTypes>
struct ExtrudeEdgesAndGenerateQuads_test : public Sofa_test<typename _DataTypes::Real>,
        ExtrudeEdgesAndGenerateQuads<_DataTypes>
{
    typedef ExtrudeEdgesAndGenerateQuads<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef unsigned int unint;


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
    void normalTests()
    {
        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        EXPECT_TRUE( m_thisObject->findData("extrudeDirection") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("thicknessIn") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("thicknessOut") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("numberOfSections") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("curveVertices") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("curveEdges") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("extrudedVertices") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("extrudedEdges") != NULL ) ;
        EXPECT_TRUE( m_thisObject->findData("extrudedQuads") != NULL ) ;

        EXPECT_NO_THROW( m_thisObject->init() ) ;
        EXPECT_NO_THROW( m_thisObject->bwdInit() ) ;
        EXPECT_NO_THROW( m_thisObject->reinit() ) ;
        EXPECT_NO_THROW( m_thisObject->reset() ) ;
        EXPECT_NO_THROW( m_thisObject->update() ) ;

        return ;
    }


    // Test size of the output mesh while varying the data numberOfSections
    void outputsSizeTest()
    {
        // Open curve
        m_thisObject->findData("curveVertices")->read("0. 0. 0.  1. 0. 0.  ");
        m_thisObject->findData("curveEdges")->read("0 1");
        m_thisObject->findData("extrudeDirection")->read("0. 0. 1.");

        m_thisObject->findData("numberOfSections")->read("1.");
        m_thisObject->update();
        EXPECT_EQ(m_thisObject->d_extrudedVertices.getValue().size(), (unint)4);
        EXPECT_EQ(m_thisObject->d_extrudedEdges.getValue().size(), (unint)4);
        EXPECT_EQ(m_thisObject->d_extrudedQuads.getValue().size(), (unint)1);

        m_thisObject->findData("numberOfSections")->read("3.");
        m_thisObject->update();
        EXPECT_EQ(m_thisObject->d_extrudedVertices.getValue().size(), (unint)8);
        EXPECT_EQ(m_thisObject->d_extrudedEdges.getValue().size(), (unint)10);
        EXPECT_EQ(m_thisObject->d_extrudedQuads.getValue().size(), (unint)3);

        m_thisObject->findData("numberOfSections")->read("0.");
        m_thisObject->update();
        EXPECT_EQ(m_thisObject->d_extrudedVertices.getValue().size(), (unint)2);
        EXPECT_EQ(m_thisObject->d_extrudedEdges.getValue().size(), (unint)1);
        EXPECT_EQ(m_thisObject->d_extrudedQuads.getValue().size(), (unint)0);


        m_thisObject->findData("numberOfSections")->read("-1.");
        m_thisObject->reinit();
        m_thisObject->update();
        EXPECT_EQ(m_thisObject->d_extrudedVertices.getValue().size(), (unint)4);
        EXPECT_EQ(m_thisObject->d_extrudedEdges.getValue().size(), (unint)4);
        EXPECT_EQ(m_thisObject->d_extrudedQuads.getValue().size(), (unint)1);


        // Closed curve
        m_thisObject->findData("curveVertices")->read("0. 0. 0.  1. 0. 0.  0. 1. 0.");
        m_thisObject->findData("curveEdges")->read("0 1 1 2 2 0");
        m_thisObject->findData("extrudeDirection")->read("0. 0. 1.");

        m_thisObject->findData("numberOfSections")->read("1.");
        m_thisObject->update();
        EXPECT_EQ(m_thisObject->d_extrudedVertices.getValue().size(), (unint)6);
        EXPECT_EQ(m_thisObject->d_extrudedEdges.getValue().size(), (unint)9);
        EXPECT_EQ(m_thisObject->d_extrudedQuads.getValue().size(), (unint)3);
    }


    // Test extrusion on a simple example
    void extrudeTest()
    {
        m_thisObject->findData("curveVertices")->read("0. 0. 0.  1. 0. 0.  ");
        m_thisObject->findData("curveEdges")->read("0 1");
        m_thisObject->findData("extrudeDirection")->read("0. 0. 1.");

        m_thisObject->findData("numberOfSections")->read("1.");
        m_thisObject->findData("thicknessIn")->read("1.");
        m_thisObject->findData("thicknessOut")->read("2.");
        m_thisObject->update();

        if(m_thisObject->d_extrudedVertices.getValue().size() != 4)
            return;

        Coord p1(0.,0.,2.);
        Coord p2(0.,0.,-1.);
        Coord p3(1.,0.,2.);
        Coord p4(1.,0.,-1.);
        Coord p;

        p = m_thisObject->d_extrudedVertices.getValue()[0];
        EXPECT_TRUE(p==p1 || p==p2 || p==p3 || p==p4);
        p = m_thisObject->d_extrudedVertices.getValue()[1];
        EXPECT_TRUE(p==p1 || p==p2 || p==p3 || p==p4);
        p = m_thisObject->d_extrudedVertices.getValue()[2];
        EXPECT_TRUE(p==p1 || p==p2 || p==p3 || p==p4);
        p = m_thisObject->d_extrudedVertices.getValue()[3];
        EXPECT_TRUE(p==p1 || p==p2 || p==p3 || p==p4);
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(ExtrudeEdgesAndGenerateQuads_test, DataTypes);

TYPED_TEST(ExtrudeEdgesAndGenerateQuads_test, NormalBehavior) {
    EXPECT_MSG_NOEMIT(Error) ;

    ASSERT_NO_THROW(this->normalTests()) ;
}

TYPED_TEST(ExtrudeEdgesAndGenerateQuads_test, OutputsSizeTest) {
    EXPECT_MSG_NOEMIT(Error) ;

    ASSERT_NO_THROW(this->outputsSizeTest()) ;
}

TYPED_TEST(ExtrudeEdgesAndGenerateQuads_test, ExtrudeTest) {
    EXPECT_MSG_NOEMIT(Error) ;

    ASSERT_NO_THROW(this->extrudeTest()) ;
}

}
