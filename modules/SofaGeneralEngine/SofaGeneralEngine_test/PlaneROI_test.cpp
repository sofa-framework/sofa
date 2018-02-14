/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaGeneralEngine/PlaneROI.h>
using sofa::component::engine::PlaneROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

using std::vector;
using std::string;


namespace sofa
{

template <typename _DataTypes>
struct PlaneROI_test : public Sofa_test<typename _DataTypes::Real>,
        PlaneROI<_DataTypes>
{
    typedef PlaneROI<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;


    Simulation* m_simu;
    Node::SPtr m_node1, m_node2;
    typename ThisClass::SPtr m_thisObject;

    void SetUp()
    {
        // SetUp1
        setSimulation(m_simu = new DAGSimulation());
        m_thisObject = New<ThisClass >();
        m_node1 = m_simu->createNewGraph("root");
        m_node1->addObject(m_thisObject);


        // SetUp2
        string scene1 =
        "<?xml version='1.0'?>"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       "
        "   <Node name='node'>                                             "
        "       <PlaneROI template='Vec3d' name='PlaneROI' plane='2 0 0  0 0 0  2 2 0  2'/> "
        "   </Node>                                                        "
        "</Node>                                                           " ;

        m_node2 = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene1.c_str(),
                                                  scene1.size()) ;
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
            "plane",
            "position", "edges",  "triangles", "tetrahedra",
            "computeEdges", "computeTriangles", "computeTetrahedra",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices",
            "pointsInROI", "edgesInROI", "trianglesInROI", "tetrahedraInROI",
            "drawBoxes", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra",
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



    /// Test isPointInPlane computation with a simple example
    void isPointInPlaneTest()
    {
        m_node2->getChild("node")->getObject("PlaneROI")->findData("position")->read("1. 0. 0. 1. 1. 0. -1 0 0");
        m_node2->getChild("node")->getObject("PlaneROI")->init();

        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("indices")->getValueString(),"0 1");
    }


    /// Test isEdgeInPlane computation with a simple example
    void isEdgeInPlaneTest()
    {
        m_node2->getChild("node")->getObject("PlaneROI")->findData("position")->read("1. 0. 0. 1. 1. 0. -1 0 0");
        m_node2->getChild("node")->getObject("PlaneROI")->findData("edges")->read("0 1 1 2");
        m_node2->getChild("node")->getObject("PlaneROI")->init();

        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("edgeIndices")->getValueString(),"0");
        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("edgesInROI")->getValueString(),"0 1");
    }


    /// Test isTriangleInPlane computation with a simple example
    void isTriangleInPlaneTest()
    {
        m_node2->getChild("node")->getObject("PlaneROI")->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 3. 0. 0.");
        m_node2->getChild("node")->getObject("PlaneROI")->findData("triangles")->read("0 1 2 1 3 2");
        m_node2->getChild("node")->getObject("PlaneROI")->init();

        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("triangleIndices")->getValueString(),"0");
        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("trianglesInROI")->getValueString(),"0 1 2");
    }


    /// Test isTetrahedraInPlane computation with a simple example
    void isTetrahedraInPlaneTest()
    {
        m_node2->getChild("node")->getObject("PlaneROI")->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 1. 0. 1. 3. 0. 0.");
        m_node2->getChild("node")->getObject("PlaneROI")->findData("tetrahedra")->read("0 1 2 3 1 2 4 3");
        m_node2->getChild("node")->getObject("PlaneROI")->init();

        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("tetrahedronIndices")->getValueString(),"0");
        EXPECT_EQ(m_node2->getChild("node")->getObject("PlaneROI")->findData("tetrahedraInROI")->getValueString(),"0 1 2 3");
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(PlaneROI_test, DataTypes);

TYPED_TEST(PlaneROI_test, attributesTests) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(PlaneROI_test, initTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->initTest()) ;
}

TYPED_TEST(PlaneROI_test, isPointInPlaneTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isPointInPlaneTest()) ;
}

TYPED_TEST(PlaneROI_test, isEdgeInPlaneTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isEdgeInPlaneTest()) ;
}

TYPED_TEST(PlaneROI_test, isTriangleInPlaneTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTriangleInPlaneTest()) ;
}

TYPED_TEST(PlaneROI_test, isTetrahedraInPlaneTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTetrahedraInPlaneTest()) ;
}

}
