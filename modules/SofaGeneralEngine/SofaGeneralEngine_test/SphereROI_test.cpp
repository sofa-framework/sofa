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

#include <SofaGeneralEngine/SphereROI.h>
using sofa::component::engine::SphereROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

using std::vector;
using std::string;


namespace sofa
{

template <typename _DataTypes>
struct SphereROI_test : public Sofa_test<typename _DataTypes::Real>,
        SphereROI<_DataTypes>
{
    typedef SphereROI<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;


    Simulation* m_simu;
    Node::SPtr m_node;
    typename ThisClass::SPtr m_thisObject;

    void SetUp()
    {
        setSimulation(m_simu = new DAGSimulation());
        m_thisObject = New<ThisClass >();
        m_node = m_simu->createNewGraph("root");
        m_node->addObject(m_thisObject);
    }



    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    void attributesTests()
    {
        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        // List of the supported attributes the user expect to find
        // This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "centers", "radii", "direction", "normal",
            "edgeAngle", "triAngle",
            "position", "edges", "quads", "triangles", "tetrahedra",
            "computeEdges", "computeTriangles", "computeQuads", "computeTetrahedra",
            "indices", "edgeIndices", "quadIndices", "triangleIndices", "tetrahedronIndices",
            "indicesOut",
            "pointsInROI", "edgesInROI", "quadsInROI", "trianglesInROI", "tetrahedraInROI",
            "drawSphere", "drawPoints", "drawEdges", "drawTriangles", "drawQuads", "drawTetrahedra",
            "drawSize"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_thisObject->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

        return ;
    }


    /// Shouldn't crash without input data
    void initTest()
    {
        EXPECT_NO_THROW(m_thisObject->init()) << "The component should succeed in being initialized.";
    }


    /// Test isPointInSphere computation with a simple example
    void isPointInSphereTest()
    {
        m_thisObject->findData("centers")->read("0. 0. 0.");
        m_thisObject->findData("radii")->read("1.");
        m_thisObject->findData("position")->read("0. 0. 0. 1. 0. 0. 2. 0. 0.");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("indices")->getValueString(),"0 1");
        EXPECT_EQ(m_thisObject->findData("indicesOut")->getValueString(),"2");
    }


    /// Test isEdgeInSphere computation with a simple example
    void isEdgeInSphereTest()
    {
        m_thisObject->findData("centers")->read("0. 0. 0.");
        m_thisObject->findData("radii")->read("1.");
        m_thisObject->findData("position")->read("0. 0. 0. 1. 0. 0. 2. 0. 0.");
        m_thisObject->findData("edges")->read("0 1 1 2");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("edgeIndices")->getValueString(),"0");
        EXPECT_EQ(m_thisObject->findData("edgesInROI")->getValueString(),"0 1");
    }


    /// Test isTriangleInSphere computation with a simple example
    void isTriangleInSphereTest()
    {
        m_thisObject->findData("centers")->read("0. 0. 0.");
        m_thisObject->findData("radii")->read("1.");
        m_thisObject->findData("position")->read("0. 0. 0. 1. 0. 0. 0. 1. 0. 2. 0. 0.");
        m_thisObject->findData("triangles")->read("0 1 2 1 3 2");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("triangleIndices")->getValueString(),"0");
        EXPECT_EQ(m_thisObject->findData("trianglesInROI")->getValueString(),"0 1 2");
    }


    /// Test isTetrahedraInSphere computation with a simple example
    void isTetrahedraInSphereTest()
    {
        m_thisObject->findData("centers")->read("0. 0. 0.");
        m_thisObject->findData("radii")->read("1.");
        m_thisObject->findData("position")->read("0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 2. 0. 0.");
        m_thisObject->findData("tetrahedra")->read("0 1 2 3 1 2 4 3");
        m_thisObject->update();

        EXPECT_EQ(m_thisObject->findData("tetrahedronIndices")->getValueString(),"0");
        EXPECT_EQ(m_thisObject->findData("tetrahedraInROI")->getValueString(),"0 1 2 3");
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(SphereROI_test, DataTypes);

TYPED_TEST(SphereROI_test, attributesTests) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->attributesTests()) ;
}


TYPED_TEST(SphereROI_test, initTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->initTest()) ;
}

TYPED_TEST(SphereROI_test, isPointInSphereTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isPointInSphereTest()) ;
}

TYPED_TEST(SphereROI_test, isEdgeInSphereTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isEdgeInSphereTest()) ;
}

TYPED_TEST(SphereROI_test, isTriangleInSphereTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTriangleInSphereTest()) ;
}

TYPED_TEST(SphereROI_test, isTetrahedraInSphereTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTetrahedraInSphereTest()) ;
}

}
