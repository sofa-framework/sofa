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
#include <SofaTest/TestMessageHandler.h>


#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/RotatableBoxROI.h>
using sofa::component::engine::RotatableBoxROI;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

using std::vector;
using std::string;


namespace sofa
{

template <typename _DataTypes>
struct RotatableBoxROI_test : public Sofa_test<typename _DataTypes::Real>,
        RotatableBoxROI<_DataTypes>
{
    typedef RotatableBoxROI<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;

    Node::SPtr m_root;
    ThisClass* m_thisObject;

    void SetUp()
    {
        // SetUp3
        string scene2 =
        "<?xml version='1.0'?>"
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >                               "
        "   <Node name='node'>                                                                     "
        "       <MeshObjLoader name='loader' filename='mesh/dragon.obj'/>                          "
        "       <Mesh name='topology' src='@loader'/>                                              "
        "       <MechanicalObject name='dofs' template='Vec3d' />                                  "
        "       <RotatableBoxROI name='RotatableBoxROI' box='-1 -1 -1 1 1 1' rotation='30 60 70'/> "
        "   </Node>                                                                                "
        "</Node>                                                                                   " ;


        m_root = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene2.c_str(),
                                                  scene2.size()) ;

        ASSERT_NE(m_root, nullptr) ;
        m_root->init(sofa::core::ExecParams::defaultInstance());

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
        m_thisObject->setName("RotatableBoxROI") ;
        EXPECT_TRUE(m_thisObject->getName() == "RotatableBoxROI") ;

        // List of the supported attributes the user expect to find
        // This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "box", "rotation",
            "position", "edges", "quad", "triangles", "tetrahedra",
            "computeEdges", "computeTriangles", "computeTetrahedra", "computeQuad", "computeHexahedra",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices", "quadIndices",
            "pointsInROI", "edgesInROI", "quadInROI", "trianglesInROI", "tetrahedraInROI",
            "drawBoxes", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra", "drawQuads", "drawHexahedra",
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
        "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >                               "
        "   <Node name='node'>                                                                     "
        "       <MeshObjLoader name='loader' filename='mesh/dragon.obj'/>                          "
        "       <Mesh name='topology' src='@loader'/>                                              "
        "       <MechanicalObject name='dofs' template='Vec3d' />                                  "
        "       <RotatableBoxROI name='RotatableBoxROI' box='-1 -1 -1 1 1 1' rotation='30 60 70'/> "
        "   </Node>                                                                                "
        "</Node>                                                                                   " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                  scene1.c_str(),
                                                  scene1.size()) ;
        ASSERT_NE(root, nullptr) ;

        root->init(sofa::core::ExecParams::defaultInstance());
        root->getChild("node")->getObject("RotatableBoxROI")->init();
        EXPECT_EQ(root->getChild("node")->getObject("RotatableBoxROI")->findData("box")->getValueString(), "-1 -1 -1 1 1 1");
    }

    /// Test isPointInRotatableBox computation with a simple example
    void isPointInRotatableBoxTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("indices")->getValueString(),"60 209 488 489 663");
    }

    /// Test isEdgeInMesh computation with a simple example
    void isEdgeInRotatableBoxTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("edgeIndices")->getValueString(),"58 528 575 576 577 838 839 840 1799 1800 3198 3199 3412 3451");
        //EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("edgesInROI")->getValueString(),"4 0 7 0 3 7 2 6 7 2 7 3 0 4 1 4 5 1 4 7 6 4 6 5 0 1 2 0 2 3");
    }


    /// Test isTriangleInMesh computation with a simple example
    void isTriangleInRotatableBoxTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("triangleIndices")->getValueString(),"200 295 697 1570 1804 1916 1930 2038 2288 2528");
        //EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("trianglesInROI")->getValueString(),"4 0 7 0 3 7 2 6 7 2 7 3 0 4 1 4 5 1 4 7 6 4 6 5 0 1 2 0 2 3");
    }

    /// Test isTetrahedraInMesh computation with a simple example
    void isTetrahedraInRotatableBoxTest()
    {
        m_root->init(sofa::core::ExecParams::defaultInstance());

        EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("tetrahedronIndices")->getValueString(),"");
        //EXPECT_EQ(m_root->getChild("node")->getObject("RotatableBoxROI")->findData("tetrahedraInROI")->getValueString(),"");
    }
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(RotatableBoxROI_test, DataTypes);

TYPED_TEST(RotatableBoxROI_test, attributesTests) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(RotatableBoxROI_test, initTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->initTest()) ;
}

TYPED_TEST(RotatableBoxROI_test, computeBoundingBoxTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->computeBoundingBoxTest()) ;
}

TYPED_TEST(RotatableBoxROI_test, isPointInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isPointInRotatableBoxTest()) ;
}

TYPED_TEST(RotatableBoxROI_test, isEdgeInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isEdgeInRotatableBoxTest()) ;
}

TYPED_TEST(RotatableBoxROI_test, isTriangleInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTriangleInRotatableBoxTest()) ;
}

TYPED_TEST(RotatableBoxROI_test, isTetrahedraInMeshTest) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_NO_THROW(this->isTetrahedraInRotatableBoxTest()) ;
}

}
