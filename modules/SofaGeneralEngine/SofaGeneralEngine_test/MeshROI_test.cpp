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

    Node::SPtr m_root;
    ThisClass* m_thisObject;

    void SetUp()
    {
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

        m_root = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene2.c_str(),
                                                  scene2.size()) ;

        ASSERT_NE(m_root, nullptr) ;

        m_thisObject = m_root->getTreeObject<ThisClass>() ;
        ASSERT_NE(m_thisObject, nullptr) ;
    }

    void TearDown()
    {
        simulation::getSimulation()->unload(m_root) ;
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    void attributesTests()
    {
        m_thisObject->setName("MeshROI") ;
        EXPECT_TRUE(m_thisObject->getName() == "MeshROI") ;

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
        string scene1 =
        "<?xml version='1.0'?>"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       "
        "   <Node name='node'>                                          "
        "       <MeshObjLoader name='loader' filename='mesh/dragon.obj'/>  "
        "       <Mesh name='topology' src='@loader'/>                      "
        "       <MeshROI template='Vec3d' name='MeshROI'/>                 "
        "   </Node>                                                        "
        "</Node>                                                           " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene1.c_str(),
                                                  scene1.size()) ;
        ASSERT_NE(root, nullptr) ;

        root->getChild("node")->getObject("MeshROI")->init();
        EXPECT_EQ(root->getChild("node")->getObject("MeshROI")->findData("box")->getValueString(),"-11.4529 -7.38909 -5.04461 11.4121 8.31288 5.01514");
    }

    /*
    //todo(dmarchal 2017-05-30) add a test demonstrating the crash.
    void checkDataConsistancy()
    {
        ////THIS SCENE IS CHRASING BECAUSE OF A SIZE MISMATCH BETWEEN topologies & positions.
        <?xml version='1.0'?>
        <Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >
           <Node name='node'>
               <MeshObjLoader name='loader' filename='mesh/cube.obj'/>
               <Mesh name='topology' src='@loader'/>
               <MeshROI template='Vec3d' name='MeshROI' position='0. 0. 0. 2. 0. 0.' />
           </Node>
        </Node>

        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("indices")->getValueString(),"0");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("indicesOut")->getValueString(),"1");
    }
    */

    /// Test isPointInMesh computation with a simple example
    void isPointInMeshTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("indices")->getValueString(),"0 3 4 5 7");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("indicesOut")->getValueString(),"1 2 6");
    }

    /// Test isEdgeInMesh computation with a simple example
    void isEdgeInMeshTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("edgeIndices")->getValueString(),"0 1 2 3 4 15");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("edgeOutIndices")->getValueString(),"5 6 7 8 9 10 11 12 13 14 16 17");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("edgesInROI")->getValueString(),"0 7 4 7 0 4 3 7 0 3 4 5");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("edgesOutROI")->getValueString(),"6 7 2 7 2 6 2 3 2 5 1 2 1 5 5 6 1 4 0 1 4 6 0 2");
    }


    /// Test isTriangleInMesh computation with a simple example
    void isTriangleInMeshTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("triangleIndices")->getValueString(), "0 1 2 3 6 7 8 9 10 11");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("triangleOutIndices")->getValueString(),"4 5");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("trianglesInROI")->getValueString(),"4 0 7 0 3 7 2 6 7 2 7 3 0 4 1 4 5 1 4 7 6 4 6 5 0 1 2 0 2 3");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("trianglesOutROI")->getValueString(),"1 5 2 5 6 2");
    }

    /// Test isTetrahedraInMesh computation with a simple example
    void isTetrahedraInMeshTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("tetrahedronIndices")->getValueString(),"");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("tetrahedronOutIndices")->getValueString(),"");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("tetrahedraInROI")->getValueString(),"");
        EXPECT_EQ(m_root->getChild("node")->getObject("MeshROI")->findData("tetrahedraOutROI")->getValueString(),"");
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(MeshROI_test, DataTypes);

TYPED_TEST(MeshROI_test, attributesTests) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(MeshROI_test, initTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->initTest()) ;
}

TYPED_TEST(MeshROI_test, computeBoundingBoxTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->computeBoundingBoxTest()) ;
}

TYPED_TEST(MeshROI_test, isPointInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isPointInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isEdgeInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isEdgeInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isTriangleInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTriangleInMeshTest()) ;
}

TYPED_TEST(MeshROI_test, isTetrahedraInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTetrahedraInMeshTest()) ;
}

}
