/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/MeshROI.h>
using sofa::component::engine::MeshROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

using std::vector;
using std::string;


namespace sofa
{

template <typename _DataTypes>
struct MeshROI_test : public Sofa_test<typename _DataTypes::Real>,
        MeshROI<_DataTypes>
{
    typedef MeshROI<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;


    Simulation* m_simu;
    Node::SPtr m_node1, m_node2, m_node3;
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
        "   <Node name='node'>                                          "
        "       <MeshObjLoader name='loader' filename='mesh/dragon.obj'/>  "
        "       <Mesh name='topology' src='@loader'/>                      "
        "       <MeshROI template='Vec3d' name='MeshROI'/>                 "
        "   </Node>                                                        "
        "</Node>                                                           " ;

        m_node2 = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene1.c_str(),
                                                  scene1.size()) ;

        // SetUp3
        string scene2 =
        "<?xml version='1.0'?>"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       "
        "   <Node name='node'>                                          "
        "       <MeshObjLoader name='loader' filename='mesh/cube.obj'/>    "
        "       <Mesh name='topology' src='@loader'/>                      "
        "       <MeshROI template='Vec3d' name='MeshROI'/>                 "
        "   </Node>                                                        "
        "</Node>                                                           " ;

        m_node3 = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene2.c_str(),
                                                  scene2.size()) ;
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
            "box",
            "position", "edges",  "triangles", "tetrahedra",
            "ROIposition", "ROIedges", "ROItriangles",
            "computeEdges", "computeTriangles", "computeTetrahedra",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices",
            "indicesOut", "edgeOutIndices", "triangleOutIndices", "tetrahedronOutIndices",
            "pointsInROI", "edgesInROI", "trianglesInROI", "tetrahedraInROI",
            "pointsOutROI", "edgesOutROI", "trianglesOutROI", "tetrahedraOutROI",
            "drawBox", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra",
            "drawSize",
            "doUpdate"
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


    /// Test bounding box computation against meshlab result
    void computeBoundingBoxTest()
    {
        m_node2->getChild("node")->getObject("MeshROI")->init();

        EXPECT_EQ(m_node2->getChild("node")->getObject("MeshROI")->findData("box")->getValueString(),"-11.4529 -7.38909 -5.04461 11.4121 8.31288 5.01514");
    }


    /// Test isPointInMesh computation with a simple example
    void isPointInMeshTest()
    {
        m_node3->getChild("node")->getObject("MeshROI")->findData("position")->read("0. 0. 0. 2. 0. 0.");
        m_node3->getChild("node")->getObject("MeshROI")->init();

        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("indices")->getValueString(),"0");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("indicesOut")->getValueString(),"1");
    }


    /// Test isEdgeInMesh computation with a simple example
    void isEdgeInMeshTest()
    {
        m_node3->getChild("node")->getObject("MeshROI")->findData("position")->read("0. 0. 0. 1. 0. 0. 2. 0. 0.");
        m_node3->getChild("node")->getObject("MeshROI")->findData("edges")->read("0 1 1 2");
        m_node3->getChild("node")->getObject("MeshROI")->init();

        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("edgeIndices")->getValueString(),"0");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("edgeOutIndices")->getValueString(),"1");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("edgesInROI")->getValueString(),"0 1");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("edgesOutROI")->getValueString(),"1 2");
    }


    /// Test isTriangleInMesh computation with a simple example
    void isTriangleInMeshTest()
    {
        m_node3->getChild("node")->getObject("MeshROI")->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 2. 0. 0.");
        m_node3->getChild("node")->getObject("MeshROI")->findData("triangles")->read("0 1 2 1 3 2");
        m_node3->getChild("node")->getObject("MeshROI")->init();

        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("triangleIndices")->getValueString(),"0");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("triangleOutIndices")->getValueString(),"1");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("trianglesInROI")->getValueString(),"0 1 2");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("trianglesOutROI")->getValueString(),"1 3 2");
    }


    /// Test isTetrahedraInMesh computation with a simple example
    void isTetrahedraInMeshTest()
    {
        m_node3->getChild("node")->getObject("MeshROI")->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 1. 0. 1. 2. 0. 0.");
        m_node3->getChild("node")->getObject("MeshROI")->findData("tetrahedra")->read("0 1 2 3 1 2 4 3");
        m_node3->getChild("node")->getObject("MeshROI")->init();

        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("tetrahedronIndices")->getValueString(),"0");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("tetrahedronOutIndices")->getValueString(),"1");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("tetrahedraInROI")->getValueString(),"0 1 2 3");
        EXPECT_EQ(m_node3->getChild("node")->getObject("MeshROI")->findData("tetrahedraOutROI")->getValueString(),"1 2 4 3");
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(MeshROI_test, DataTypes);

TYPED_TEST(MeshROI_test, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(MeshROI_test, initTest) {
    ASSERT_NO_THROW(this->initTest()) ;
}

TYPED_TEST(MeshROI_test, computeBoundingBoxTest) {
    ASSERT_NO_THROW(this->computeBoundingBoxTest()) ;
}

TYPED_TEST(MeshROI_test, isPointInMeshTest) {
    ASSERT_NO_THROW(this->isPointInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isEdgeInMeshTest) {
    ASSERT_NO_THROW(this->isEdgeInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isTriangleInMeshTest) {
    ASSERT_NO_THROW(this->isTriangleInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isTetrahedraInMeshTest) {
    ASSERT_NO_THROW(this->isTetrahedraInMeshTest()) ;
}

}
