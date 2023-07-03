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

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <gtest/gtest.h>
using ::testing::Types;
#include <sofa/core/fwd.h>
#include <sofa/helper/BackTrace.h>

#include <sofa/core/objectmodel/ComponentState.h>
using sofa::core::objectmodel::ComponentState;

#include <sofa/component/engine/select/BoxROI.h>
using sofa::component::engine::select::BoxROI;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation;
using sofa::simulation::graph::DAGSimulation;
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;
using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::New;
using sofa::defaulttype::Vec3Types;

using namespace sofa::type;
using namespace sofa::defaulttype;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML;

#include <sofa/helper/logging/Message.h>
using sofa::helper::logging::MessageDispatcher;

#include <sofa/testing/TestMessageHandler.h>
#include <sofa/testing/BaseTest.h>

#include <sofa/simulation/graph/SimpleApi.h>

template <typename TDataType>
struct BoxROITest :  public sofa::testing::BaseTest
{
    typedef BoxROI<TDataType> TheBoxROI;
    Simulation* m_simu  {nullptr};
    Node::SPtr m_root;
    Node::SPtr m_node;
    typename TheBoxROI::SPtr m_boxroi;

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.Engine.Select");

        m_simu = sofa::simulation::getSimulation();
        ASSERT_NE(m_simu, nullptr);

        m_root = m_simu->createNewGraph("root");

        m_node = m_root->createChild("node");
        m_boxroi = New< TheBoxROI >();
        m_node->addObject(m_boxroi);
    }

    void TearDown() override
    {
        if (m_root != nullptr){
            sofa::simulation::node::unload(m_root);
        }
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    void attributesTests(){

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        std::vector<string> attrnames = {
            "box", "orientedBox",
            "position", "edges",  "triangles", "tetrahedra", "hexahedra", "quad",
            "computeEdges", "computeTriangles", "computeTetrahedra", "computeHexahedra", "computeQuad",
            "indices", "edgeIndices", "triangleIndices", "tetrahedronIndices", "hexahedronIndices",
            "quadIndices",
            "pointsInROI", "edgesInROI", "trianglesInROI", "tetrahedraInROI", "hexahedraInROI", "quadInROI",
            "nbIndices",
            "drawBoxes", "drawPoints", "drawEdges", "drawTriangles", "drawTetrahedra", "drawHexahedra", "drawQuads",
            "drawSize",
            "doUpdate"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_boxroi->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'.";

        return;
    }

    void checkGracefullHandlingOfInvalidUsage(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node name='Root' gravity='0 0 0' time='0' animate='0'>       "
                "   <Node name='Level 1'>                                      "
                "       <BoxROI name='myBoxROI'/>                              "
                "   </Node>                                                    "
                "</Node>                                                       ";

        EXPECT_MSG_EMIT(Error); // Unable to find a MechanicalObject for this component.
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.c_str());
        EXPECT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        BaseObject* boxroi = root->getTreeNode("Level 1")->getObject("myBoxROI");
        EXPECT_NE(boxroi, nullptr);

        EXPECT_EQ(boxroi->getComponentState(), ComponentState::Invalid ) << "The component cannot be initialized because it is missing a MechanicalObject. "
                                                                            "But it shouldn't crash or segfault. ";

        boxroi->reinit();

        EXPECT_EQ(boxroi->getComponentState(), ComponentState::Invalid ) << "Reinit shouln't crash or change the state because there is still no MechanicalObject. ";

    }


    void checkAutomaticSearchingOfMechanicalObject(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node name='Root' gravity='0 0 0' time='0' animate='0'>       "
                "   <Node name='Level 1'>                                      "
                "       <TriangleSetTopologyContainer name='Container'/>       "
                "       <MechanicalObject name='meca' position='0 0 0 1 1 1'/> "
                "       <BoxROI name='myBoxROI'/>                              "
                "   </Node>                                                    "
                "</Node>                                                       ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.c_str());
        EXPECT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheBoxROI* boxroi = root->getTreeObject<TheBoxROI>();
        EXPECT_NE(boxroi, nullptr);
        EXPECT_EQ(boxroi->getComponentState(), ComponentState::Valid ) << "The component should succeed in being initialized because there is a MechanicalObject in the current context. ";
    }


    void checkAutomaticSearchingOfMechanicalObjectParent(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >   "
                "   <MechanicalObject name='meca' position='0 0 0 1 1 1'/>     "
                "   <Node name='Level 1'>                                      "
                "       <TriangleSetTopologyContainer  name='Container' />     "
                "       <BoxROI name='myBoxROI'/>                              "
                "   </Node>                                                    "
                "</Node>                                                       ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene.c_str());
        EXPECT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        TheBoxROI* boxroi = root->getTreeObject<TheBoxROI>();
        EXPECT_NE(boxroi, nullptr);
        EXPECT_EQ(boxroi->getComponentState(), ComponentState::Valid ) << "The component should succeed in being initialized because there is a MechanicalObject in the current context. ";
    }

    void checkAutomaticSearchingOfMeshLoader(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >   "
                "   <Node name='Level 1'>                                      "
                "       <RequiredPlugin  name='Sofa.Component.IO.Mesh' />      "
                "       <TriangleSetTopologyContainer  name='Container' />     "
                "       <MeshOBJLoader filename='mesh/single_triangle.obj'/>                   "
                "       <BoxROI name='myBoxROI'/>                              "
                "   </Node>                                                    "
                "</Node>                                                       ";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.c_str());
        EXPECT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());
        BaseObject* boxroi = root->getTreeNode("Level 1")->getObject("myBoxROI");

        EXPECT_NE(boxroi, nullptr);
        EXPECT_EQ(boxroi->getComponentState(), ComponentState::Valid ) << "The component should succeed in being initialized because there is a MeshLoader and a TopologyContainer in the current context. ";
    }


    /// Test isPointInBox computation with a simple example
    void isPointInBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 1. 1. 1.");
        m_boxroi->findData("position")->read("0. 0. 0. 1. 1. 1. 2. 2. 2.");
        m_boxroi->update();

        EXPECT_EQ(m_boxroi->findData("indices")->getValueString(),"0 1");
    }


    /// Test isEdgeInBox computation with a simple example
    void isEdgeInBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 1. 1. 1.");
        m_boxroi->findData("position")->read("0. 0. 0. 1. 0. 0. 2. 0. 0.");
        m_boxroi->findData("edges")->read("0 1 1 2");
        m_boxroi->update();

        EXPECT_EQ(m_boxroi->findData("edgeIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("edgesInROI")->getValueString(),"0 1");
    }


    /// Test isTriangleInBox computation with a simple example
    void isTriangleInBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 1. 1. 1.");
        m_boxroi->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 2. 0. 0.");
        m_boxroi->findData("triangles")->read("0 1 2 1 3 2");
        m_boxroi->update();

        EXPECT_EQ(m_boxroi->findData("triangleIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("trianglesInROI")->getValueString(),"0 1 2");
    }


    /// Test isTetrahedraInBox computation with a simple example
    void isTetrahedraInBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 1. 1. 1.");
        m_boxroi->findData("position")->read("0. 0. 0. 1. 0. 0. 1. 1. 0. 1. 0. 1. 2. 0. 0.");
        m_boxroi->findData("tetrahedra")->read("0 1 2 3 1 2 4 3");
        m_boxroi->update();

        EXPECT_EQ(m_boxroi->findData("tetrahedronIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("tetrahedraInROI")->getValueString(),"0 1 2 3");
    }


    /// Test isPointInOrientedBox computation with a simple example
    void isPointInOrientedBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 0. 0. 0.");
        m_boxroi->findData("orientedBox")->read("2 0 0  0 0 0  2 2 2 2");
        m_boxroi->findData("position")->read("1. 0. 0.   1. 0. 1.   0. 0. 1.");
        m_boxroi->init();

        EXPECT_EQ(m_boxroi->findData("indices")->getValueString(),"0 1");
    }


    /// Test isEdgeInOrientedBox computation with a simple example
    void isEdgeInOrientedBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 0. 0. 0.");
        m_boxroi->findData("orientedBox")->read("2 0 0  0 0 0  2 2 2  2");
        m_boxroi->findData("position")->read("0. 0. 0.   1. 0. 1.   0. 0. 1.");
        m_boxroi->findData("edges")->read("0 1 0 2");
        m_boxroi->init();

        EXPECT_EQ(m_boxroi->findData("edgeIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("edgesInROI")->getValueString(),"0 1");
    }


    /// Test isTriangleInOrientedBox computation with a simple example
    void isTriangleInOrientedBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 0. 0. 0.");
        m_boxroi->findData("orientedBox")->read("2 0 0  0 0 0  2 2 2  2");
        m_boxroi->findData("position")->read("0. 0. 0.   1. 0. 0.   1. 1. 0.  0. 0. -1.");
        m_boxroi->findData("triangles")->read("0 1 2 0 3 1");
        m_boxroi->init();

        EXPECT_EQ(m_boxroi->findData("triangleIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("trianglesInROI")->getValueString(),"0 1 2");
    }


    /// Test isTetrahedraInOrientedBox computation with a simple example
    void isTetrahedraInOrientedBoxTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 0. 0. 0.");
        m_boxroi->findData("orientedBox")->read("2 0 0  0 0 0  2 2 2 2");
        m_boxroi->findData("position")->read("0. 0. 0.   1. 0. 0.    1. 1. 0.   1. 0. 1.   0. 0. -2.");
        m_boxroi->findData("tetrahedra")->read("0 1 2 3 0 1 2 4");
        m_boxroi->init();

        EXPECT_EQ(m_boxroi->findData("tetrahedronIndices")->getValueString(),"0");
        EXPECT_EQ(m_boxroi->findData("tetrahedraInROI")->getValueString(),"0 1 2 3");
    }


    /// Test isTetrahedraInOrientedBox computation with a simple example
    void isPointInBoxesTest()
    {
        m_boxroi->findData("box")->read("0. 0. 0. 1. 1. 1.  0. 0. 0. -1 -1 -1");
        m_boxroi->findData("orientedBox")->read("3 0 0  1 0 0  3 2 2  2    -3 0 0  -1 0 0  -3 -2 -2  2");
        m_boxroi->findData("position")->read("1. 0. 0.   -1. 0. 0.   2. 0. 0.   -2. 0. 0.  1. -1. 0.  -1. 1. 0.");
        m_boxroi->init();

        EXPECT_EQ(m_boxroi->findData("indices")->getValueString(),"0 1 2 3");
    }


    /// Test computeBBox computation with a simple example
    void computeBBoxTest()
    {
        m_boxroi->findData("box")->read("-1. -1. -1.  0. 0. 0.   1. 1. 1.  2. 2. 2.");
        m_boxroi->computeBBox(nullptr, false);

        EXPECT_EQ(m_boxroi->f_bbox.getValue().minBBox(), Vec3(-1,-1,-1));
        EXPECT_EQ(m_boxroi->f_bbox.getValue().maxBBox(), Vec3(2,2,2));

        m_boxroi->findData("box")->read("-1. -1. -1.  0. 0. 0.");
        m_boxroi->findData("orientedBox")->read("0 0 0  2 0 0  2 2 0 2");
        m_boxroi->computeBBox(nullptr, false);

        EXPECT_EQ(m_boxroi->f_bbox.getValue().minBBox(), Vec3(-1,-1,-1));
        EXPECT_EQ(m_boxroi->f_bbox.getValue().maxBBox(), Vec3(2,2,1));
    }

};


//TODO(dmarchal): There is a problem of segfault with the test & Rigid3 types.
//Please fix this either the tests or the BoxROI implementation
typedef Types<
    Vec3Types
    //,Rigid3dTypes

> DataTypes;

TYPED_TEST_SUITE(BoxROITest, DataTypes);


TYPED_TEST(BoxROITest, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests());
}

TYPED_TEST(BoxROITest, checkGracefullHandlingOfInvalidUsage) {
    ASSERT_NO_THROW(this->checkGracefullHandlingOfInvalidUsage());
}

TYPED_TEST(BoxROITest, checkAutomaticSearchingOfMechanicalObject) {
    ASSERT_NO_THROW(this->checkAutomaticSearchingOfMechanicalObject());
}

TYPED_TEST(BoxROITest, checkAutomaticSearchingOfMechanicalObjectParent) {
    ASSERT_NO_THROW(this->checkAutomaticSearchingOfMechanicalObject());
}

TYPED_TEST(BoxROITest, checkAutomaticSearchingOfMeshLoader) {
    ASSERT_NO_THROW(this->checkAutomaticSearchingOfMeshLoader());
}

TYPED_TEST(BoxROITest, isPointInBoxTest) {
    ASSERT_NO_THROW(this->isPointInBoxTest());
}

TYPED_TEST(BoxROITest, isEdgeInBoxTest) {
    ASSERT_NO_THROW(this->isEdgeInBoxTest());
}

TYPED_TEST(BoxROITest, isTriangleInBoxTest) {
    ASSERT_NO_THROW(this->isTriangleInBoxTest());
}

TYPED_TEST(BoxROITest, isTetrahedraInBoxTest) {
    ASSERT_NO_THROW(this->isTetrahedraInBoxTest());
}


TYPED_TEST(BoxROITest, isPointInOrientedBoxTest) {
    ASSERT_NO_THROW(this->isPointInOrientedBoxTest());
}

TYPED_TEST(BoxROITest, isEdgeInOrientedBoxTest) {
    ASSERT_NO_THROW(this->isEdgeInOrientedBoxTest());
}

TYPED_TEST(BoxROITest, isTriangleInOrientedBoxTest) {
    ASSERT_NO_THROW(this->isTriangleInOrientedBoxTest());
}

TYPED_TEST(BoxROITest, isTetrahedraInOrientedBoxTest) {
    ASSERT_NO_THROW(this->isTetrahedraInOrientedBoxTest());
}

TYPED_TEST(BoxROITest, isPointInBoxesTest) {
    ASSERT_NO_THROW(this->isPointInBoxesTest());
}

TYPED_TEST(BoxROITest, computeBBoxTest) {
    ASSERT_NO_THROW(this->computeBBoxTest());
}

