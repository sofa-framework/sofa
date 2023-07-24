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
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/utility/TopologyChecker.h>

#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/simulation/Node.h>

using sofa::testing::BaseSimulationTest;

namespace 
{

using namespace sofa::component::topology;
using namespace sofa::component::topology::container::dynamic;
using namespace sofa::core::topology;
using namespace sofa::simulation;
using sofa::component::topology::utility::TopologyChecker;

/**  Test TopologyChecker class on different valid and unvalid topology containers
  */

struct TopologyChecker_test: public BaseSimulationTest
{
    /// Store SceneInstance 
    BaseSimulationTest::SceneInstance m_instance;

    /// Name of the file to load
    std::string m_fileName = "";

    /// Node containing topology
    Node::SPtr m_topoNode = nullptr;

    /// Method use at start to load the scene file    
    void SetUp() override
    {
        // Load the scene from the xml file
        const std::string filePath = std::string(SOFA_COMPONENT_TOPOLOGY_UTILITY_TEST_SCENES_DIR) + "/" + m_fileName;
        m_instance = BaseSimulationTest::SceneInstance();
        // Load scene
        m_instance.loadSceneFile(filePath);
        // Init scene
        m_instance.initScene();

        // Test if root is not null
        if(!m_instance.root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return;
        }
    }

    /// Method to really do the test per type of topology container, to be implemented by child classes
    virtual bool testValidTopology() = 0;

    /// Method to test invalid topology container, to be implemented by child classes
    virtual bool testInvalidContainer() = 0;

    /// Unload the scene
    void TearDown() override
    {
        if (m_instance.root !=nullptr)
            sofa::simulation::node::unload(m_instance.root);
    }

};


struct EdgeTopologyChecker_test : TopologyChecker_test
{
    EdgeTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "/RemovingTriangle2EdgeProcess.scn";
    }


    EdgeSetTopologyContainer* loadTopo()
    {
        const Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }

        const Node::SPtr nodeTopoTri = root.get()->getChild("SquareGravity");
        if (!nodeTopoTri)
        {
            ADD_FAILURE() << "Error 'SquareGravity' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        m_topoNode = nodeTopoTri->getChild("Edge Mesh");
        if (!m_topoNode)
        {
            ADD_FAILURE() << "Error 'Edge Mesh' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        EdgeSetTopologyContainer* topoCon = dynamic_cast<EdgeSetTopologyContainer*>(m_topoNode->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: EdgeSetTopologyContainer not found in 'Edge Mesh' Node, in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        return topoCon;
    }


    bool testUnsetTopologyLink()
    {
        const EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"}});

        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        return checker->l_topology.get() == topoCon;
    }


    bool testValidTopology() override
    {
        EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 96);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        EXPECT_EQ(checker->checkTopology(), true);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 333);
        EXPECT_EQ(topoCon->getNbPoints(), 261);

        return checker->checkTopology() == true;
    }


    bool testInvalidContainer() override
    {
        EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);
        
        // mix edge
        edges[0][0] = edges[0][1];

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkEdgeTopology() == false;
    }


    bool testInvalidEdge2VertexContainer()
    {
        EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);

        // Add edges without updating cross container
        edges.push_back(Topology::Edge(0, 10));
        EXPECT_EQ(topoCon->getNbEdges(), 97);

        // container should be ok
        EXPECT_EQ(checker->checkEdgeContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        return checker->checkEdgeToVertexCrossContainer() == false;
    }
};


struct TriangleTopologyChecker_test : TopologyChecker_test
{
    TriangleTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "/RemovingTrianglesProcess.scn";
    }


    TriangleSetTopologyContainer* loadTopo()
    {
        const Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }

        m_topoNode = root.get()->getChild("SquareGravity");
        if (!m_topoNode)
        {
            ADD_FAILURE() << "Error 'SquareGravity' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        TriangleSetTopologyContainer* topoCon = dynamic_cast<TriangleSetTopologyContainer*>(m_topoNode->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TriangleSetTopologyContainer not found in 'SquareGravity' Node, in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        return topoCon;
    }


    bool testValidTopology() override
    {
        TriangleSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbTriangles(), 1450);
        EXPECT_EQ(topoCon->getNbEdges(), 2223);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        EXPECT_EQ(checker->checkTopology(), true);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 145);
        EXPECT_EQ(topoCon->getNbEdges(), 384);
        EXPECT_EQ(topoCon->getNbPoints(), 261);

        return checker->checkTopology() == true;
    }

    bool testInvalidContainer() override
    {
        TriangleSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto triangles = sofa::helper::getWriteAccessor(topoCon->d_triangle);

        // mix triangle
        triangles[0][0] = triangles[0][1];

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkTriangleTopology() == false;
    }

    
    bool testInvalidTriangle2EdgeContainer()
    {
        TriangleSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto triangles = sofa::helper::getWriteAccessor(topoCon->d_triangle);
        
        // Add triangle without updating cross container
        triangles.push_back(Topology::Triangle(0, 10, 100));

        // container should be ok
        EXPECT_EQ(checker->checkTriangleContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), false);
        // restore good container
        triangles.resize(triangles.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), true);


        // Mix some triangle vertices
        const int tmp = triangles[0][0];
        triangles[0][0] = triangles[10][0];
        triangles[10][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), false);
        // restore good container
        triangles[10][0] = triangles[0][0];
        triangles[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), true);

        // Mix some triangles
        const Topology::Triangle tmpT = triangles[0];
        triangles[0] = triangles[10];
        triangles[10] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), false);
        // restore good container
        triangles[10] = triangles[0];
        triangles[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTriangleToEdgeCrossContainer(), true);

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);
        const Topology::Edge tmpE = edges[0];
        edges[0] = edges[10];
        edges[10] = tmpE;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTopology(), false);

        return true;
    }


    bool testInvalidTriangle2VertexContainer()
    {
        TriangleSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto triangles = sofa::helper::getWriteAccessor(topoCon->d_triangle);

        // Add triangle without updating cross container
        triangles.push_back(Topology::Triangle(0, 10, 100));

        // container should be ok
        EXPECT_EQ(checker->checkTriangleContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), false);
        // restore good container
        triangles.resize(triangles.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), true);


        // Mix some triangle vertices
        const int tmp = triangles[0][0];
        triangles[0][0] = triangles[10][0];
        triangles[10][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), false);
        // restore good container
        triangles[10][0] = triangles[0][0];
        triangles[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), true);

        // Mix some triangles
        const Topology::Triangle tmpT = triangles[0];
        triangles[0] = triangles[10];
        triangles[10] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), false);
        // restore good container
        triangles[10] = triangles[0];
        triangles[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTriangleToVertexCrossContainer(), true);

        return true;
    }
};


struct QuadTopologyChecker_test : TopologyChecker_test
{
    QuadTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "/RemovingQuad2TriangleProcess.scn";
    }


    QuadSetTopologyContainer* loadTopo()
    {
        const Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }

        m_topoNode = root.get()->getChild("Q");
        if (!m_topoNode)
        {
            ADD_FAILURE() << "Error 'Q' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(m_topoNode->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: QuadSetTopologyContainer not found in 'Q' Node, in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        return topoCon;
    }


    bool testValidTopology() override
    {
        QuadSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbQuads(), 9);
        EXPECT_EQ(topoCon->getNbEdges(), 24);
        EXPECT_EQ(topoCon->getNbPoints(), 16);

        EXPECT_EQ(checker->checkTopology(), true);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 40; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbQuads(), 7);
        EXPECT_EQ(topoCon->getNbEdges(), 20);
        EXPECT_EQ(topoCon->getNbPoints(), 14);

        return checker->checkTopology() == true;
    }


    bool testInvalidContainer() override
    {
        QuadSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto quads = sofa::helper::getWriteAccessor(topoCon->d_quad);
        
        // mix triangle
        quads[0][0] = quads[0][1];

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkQuadTopology() == false;
    }


    bool testInvalidQuad2EdgeContainer()
    {
        QuadSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto quads = sofa::helper::getWriteAccessor(topoCon->d_quad);

        // Add triangle without updating cross container
        quads.push_back(Topology::Quad(0, 2, 4, 8));

        // container should be ok
        EXPECT_EQ(checker->checkQuadContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), false);
        // restore good container
        quads.resize(quads.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), true);


        // Mix some quad vertices
        const int tmp = quads[0][0];
        quads[0][0] = quads[4][0];
        quads[4][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), false);
        // restore good container
        quads[4][0] = quads[0][0];
        quads[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), true);

        // Mix some quads
        const Topology::Quad tmpT = quads[0];
        quads[0] = quads[4];
        quads[4] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), false);
        // restore good container
        quads[4] = quads[0];
        quads[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkQuadToEdgeCrossContainer(), true);

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);
        const Topology::Edge tmpE = edges[0];
        edges[0] = edges[10];
        edges[10] = tmpE;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTopology(), false);

        return true;
    }


    bool testInvalidQuad2VertexContainer()
    {
        QuadSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto quads = sofa::helper::getWriteAccessor(topoCon->d_quad);

        // Add triangle without updating cross container
        quads.push_back(Topology::Quad(0, 2, 4, 8));

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), false);
        // restore good container
        quads.resize(quads.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), true);


        // Mix some quad vertices
        const int tmp = quads[0][0];
        quads[0][0] = quads[4][0];
        quads[4][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), false);
        // restore good container
        quads[4][0] = quads[0][0];
        quads[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), true);

        // Mix some quads
        const Topology::Quad tmpT = quads[0];
        quads[0] = quads[4];
        quads[4] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), false);
        // restore good container
        quads[4] = quads[0];
        quads[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkQuadToVertexCrossContainer(), true);

        return true;
    }
};



struct TetrahedronTopologyChecker_test : TopologyChecker_test
{
    TetrahedronTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "/RemovingTetraProcess.scn";
    }


    TetrahedronSetTopologyContainer* loadTopo()
    {
        const Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }

        m_topoNode = root.get()->getChild("TT");
        if (!m_topoNode)
        {
            ADD_FAILURE() << "Error 'TT' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(m_topoNode->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TetrahedronSetTopologyContainer not found in 'TT' Node, in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        return topoCon;
    }


    bool testValidTopology() override
    {
        TetrahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbTetrahedra(), 44);
        EXPECT_EQ(topoCon->getNbTriangles(), 112);
        EXPECT_EQ(topoCon->getNbEdges(), 93);
        EXPECT_EQ(topoCon->getNbPoints(), 26);

        EXPECT_EQ(checker->checkTopology(), true);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTetrahedra(), 34);
        EXPECT_EQ(topoCon->getNbTriangles(), 97);
        EXPECT_EQ(topoCon->getNbEdges(), 86);
        EXPECT_EQ(topoCon->getNbPoints(), 25);

        return checker->checkTopology() == true;
    }

    bool testInvalidContainer() override
    {
        TetrahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto tetra = sofa::helper::getWriteAccessor(topoCon->d_tetrahedron);
        
        // mix triangle
        tetra[0][0] = tetra[0][1];

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkTetrahedronTopology() == false;
    }

    bool testInvalidTetrahedron2TriangleContainer()
    {
        TetrahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                {"name", "checker"},
                                {"topology", topoLink}
            });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto tetra = sofa::helper::getWriteAccessor(topoCon->d_tetrahedron);

        // Add triangle without updating cross container
        tetra.push_back(Topology::Tetrahedron(0, 2, 8, 12));

        // container should be ok
        EXPECT_EQ(checker->checkTetrahedronContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), false);
        // restore good container
        tetra.resize(tetra.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), true);

        // Mix some tetrahedron vertices
        int tmp = tetra[0][0];
        tetra[0][0] = tetra[10][0];
        tetra[10][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), false);
        // restore good container
        tetra[10][0] = tetra[0][0];
        tetra[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), true);

        // Mix some triangles
        Topology::Tetrahedron tmpT = tetra[0];
        tetra[0] = tetra[10];
        tetra[10] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), false);
        // restore good container
        tetra[10] = tetra[0];
        tetra[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToTriangleCrossContainer(), true);

        auto triangles = sofa::helper::getWriteAccessor(topoCon->d_triangle);
        // Mix some triangles
        Topology::Triangle tmpTri = triangles[0];
        triangles[0] = triangles[10];
        triangles[10] = tmpTri;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTopology(), false);
        // restore good container
        triangles[10] = triangles[0];
        triangles[0] = tmpTri;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTopology(), true);

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);
        Topology::Edge tmpE = edges[0];
        edges[0] = edges[10];
        edges[10] = tmpE;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTopology(), false);

        return true;
    }

    bool testInvalidTetrahedron2EdgeContainer()
    {
        TetrahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto tetra = sofa::helper::getWriteAccessor(topoCon->d_tetrahedron);

        // Add triangle without updating cross container
        tetra.push_back(Topology::Tetrahedron(0, 2, 8, 12));

        // Topology checker should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), false);
        // restore good container
        tetra.resize(tetra.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), true);

        // Mix some tetrahedron vertices
        const int tmp = tetra[0][0];
        tetra[0][0] = tetra[10][0];
        tetra[10][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), false);
        // restore good container
        tetra[10][0] = tetra[0][0];
        tetra[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), true);

        // Mix some triangles
        const Topology::Tetrahedron tmpT = tetra[0];
        tetra[0] = tetra[10];
        tetra[10] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), false);
        // restore good container
        tetra[10] = tetra[0];
        tetra[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToEdgeCrossContainer(), true);

        return true;
    }

    bool testInvalidTetrahedron2VertexContainer()
    {
        TetrahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto tetra = sofa::helper::getWriteAccessor(topoCon->d_tetrahedron);

        // Add triangle without updating cross container
        tetra.push_back(Topology::Tetrahedron(0, 2, 8, 12));

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), false);
        // restore good container
        tetra.resize(tetra.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), true);

        // Mix some tetrahedron vertices
        const int tmp = tetra[0][0];
        tetra[0][0] = tetra[10][0];
        tetra[10][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), false);
        // restore good container
        tetra[10][0] = tetra[0][0];
        tetra[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), true);

        // Mix some triangles
        const Topology::Tetrahedron tmpT = tetra[0];
        tetra[0] = tetra[10];
        tetra[10] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), false);
        // restore good container
        tetra[10] = tetra[0];
        tetra[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkTetrahedronToVertexCrossContainer(), true);

        return true;
    }
};



struct HexahedronTopologyChecker_test : TopologyChecker_test
{
    HexahedronTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "/RemovingHexa2QuadProcess.scn";
    }


    HexahedronSetTopologyContainer* loadTopo()
    {
        const Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }

        m_topoNode = root.get()->getChild("H");
        if (!m_topoNode)
        {
            ADD_FAILURE() << "Error 'H' Node not found in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(m_topoNode->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: HexahedronSetTopologyContainer not found in 'H' Node, in scene: " << m_fileName << std::endl;
            return nullptr;
        }

        return topoCon;
    }


    bool testValidTopology() override
    {
        HexahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbHexahedra(), 9);
        EXPECT_EQ(topoCon->getNbQuads(), 42);
        EXPECT_EQ(topoCon->getNbEdges(), 64);
        EXPECT_EQ(topoCon->getNbPoints(), 32);

        EXPECT_EQ(checker->checkTopology(), true);

        // to test incise animates the scene at least 1.2s
        for (int i = 0; i < 41; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbHexahedra(), 5);
        EXPECT_EQ(topoCon->getNbQuads(), 25);
        EXPECT_EQ(topoCon->getNbEdges(), 41);
        EXPECT_EQ(topoCon->getNbPoints(), 22);

        return checker->checkTopology() == true;
    }

    bool testInvalidContainer() override
    {
        HexahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        auto hexahedra = sofa::helper::getWriteAccessor(topoCon->d_hexahedron);
        
        // mix triangle
        hexahedra[0][0] = hexahedra[0][1];

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkHexahedronTopology() == false;
    }

    bool testInvalidHexahedron2QuadContainer()
    {
        HexahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<sofa::type::vector<Topology::Hexahedron> > > hexa = topoCon->d_hexahedron;

        // Add triangle without updating cross container
        hexa.push_back(Topology::Hexahedron(0, 2, 4, 12, 14, 18, 20, 22));

        // container should be ok
        EXPECT_EQ(checker->checkHexahedronContainer(), true);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), false);
        // restore good container
        hexa.resize(hexa.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), true);

        // Mix some hexahedron vertices
        const int tmp = hexa[0][0];
        hexa[0][0] = hexa[6][0];
        hexa[6][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), false);
        // restore good container
        hexa[6][0] = hexa[0][0];
        hexa[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), true);

        // Mix some triangles
        const Topology::Hexahedron tmpT = hexa[0];
        hexa[0] = hexa[6];
        hexa[6] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), false);
        // restore good container
        hexa[6] = hexa[0];
        hexa[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), true);

        auto quads = sofa::helper::getWriteAccessor(topoCon->d_quad);
        // Mix some quads
        const Topology::Quad tmpTri = quads[0];
        quads[0] = quads[10];
        quads[10] = tmpTri;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), false);
        // restore good container
        quads[10] = quads[0];
        quads[0] = tmpTri;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToQuadCrossContainer(), true);

        return true;
    }

    bool testInvalidHexahedron2EdgeContainer()
    {
        HexahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<sofa::type::vector<Topology::Hexahedron> > > hexa = topoCon->d_hexahedron;

        // Add triangle without updating cross container
        hexa.push_back(Topology::Hexahedron(0, 2, 4, 12, 14, 18, 20, 22));

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), false);
        // restore good container
        hexa.resize(hexa.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), true);

        // Mix some hexahedron vertices
        const int tmp = hexa[0][0];
        hexa[0][0] = hexa[6][0];
        hexa[6][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), false);
        // restore good container
        hexa[6][0] = hexa[0][0];
        hexa[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), true);

        // Mix some triangles
        const Topology::Hexahedron tmpT = hexa[0];
        hexa[0] = hexa[6];
        hexa[6] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), false);
        // restore good container
        hexa[6] = hexa[0];
        hexa[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), true);

        auto edges = sofa::helper::getWriteAccessor(topoCon->d_edge);
        const Topology::Edge tmpE = edges[0];
        edges[0] = edges[10];
        edges[10] = tmpE;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToEdgeCrossContainer(), false);

        return true;
    }

    bool testInvalidHexahedron2VertexContainer()
    {
        HexahedronSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        const auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                                           {"name", "checker"},
                                                           {"topology", topoLink}
                                                       });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<sofa::type::vector<Topology::Hexahedron> > > hexa = topoCon->d_hexahedron;

        // Add triangle without updating cross container
        hexa.push_back(Topology::Hexahedron(0, 2, 4, 12, 14, 18, 20, 22));

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), false);
        // restore good container
        hexa.resize(hexa.size() - 1);
        EXPECT_MSG_NOEMIT(Error); // reset no emit error for next true negative test
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), true);

        // Mix some hexahedron vertices
        const int tmp = hexa[0][0];
        hexa[0][0] = hexa[6][0];
        hexa[6][0] = tmp;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), false);
        // restore good container
        hexa[6][0] = hexa[0][0];
        hexa[0][0] = tmp;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), true);

        // Mix some triangles
        const Topology::Hexahedron tmpT = hexa[0];
        hexa[0] = hexa[6];
        hexa[6] = tmpT;
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), false);
        // restore good container
        hexa[6] = hexa[0];
        hexa[0] = tmpT;
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_EQ(checker->checkHexahedronToVertexCrossContainer(), true);

        return true;
    }
};



TEST_F(EdgeTopologyChecker_test, UnsetTopologyLink)
{
    ASSERT_TRUE(this->testUnsetTopologyLink());
}

TEST_F(EdgeTopologyChecker_test, Check_Valid_Topology)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(EdgeTopologyChecker_test, Check_Invalid_Container)
{
    ASSERT_TRUE(this->testInvalidContainer());
}

TEST_F(EdgeTopologyChecker_test, Check_Invalid_Edge2VertexCrossContainer)
{
    ASSERT_TRUE(this->testInvalidEdge2VertexContainer());
}



TEST_F(TriangleTopologyChecker_test, Check_Valid_Topology)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(TriangleTopologyChecker_test, Check_Invalid_Containers)
{
    ASSERT_TRUE(this->testInvalidContainer());
}

TEST_F(TriangleTopologyChecker_test, Check_Invalid_Triangle2EdgeCrossContainer)
{
    ASSERT_TRUE(this->testInvalidTriangle2EdgeContainer());
}

TEST_F(TriangleTopologyChecker_test, Check_Invalid_Triangle2VertexCrossContainer)
{
    ASSERT_TRUE(this->testInvalidTriangle2VertexContainer());
}



TEST_F(QuadTopologyChecker_test, Check_Valid_Topology)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(QuadTopologyChecker_test, Check_Invalid_Containers)
{
    ASSERT_TRUE(this->testInvalidContainer());
}

TEST_F(QuadTopologyChecker_test, Check_Invalid_Quad2EdgeCrossContainer)
{
    ASSERT_TRUE(this->testInvalidQuad2EdgeContainer());
}

TEST_F(QuadTopologyChecker_test, Check_Invalid_Quad2VertexCrossContainer)
{
    ASSERT_TRUE(this->testInvalidQuad2VertexContainer());
}



TEST_F(TetrahedronTopologyChecker_test, Check_Valid_Topology)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(TetrahedronTopologyChecker_test, Check_Invalid_Containers)
{
    ASSERT_TRUE(this->testInvalidContainer());
}

TEST_F(TetrahedronTopologyChecker_test, Check_Invalid_Tetrahedron2TriangleCrossContainer)
{
    ASSERT_TRUE(this->testInvalidTetrahedron2TriangleContainer());
}

TEST_F(TetrahedronTopologyChecker_test, Check_Invalid_Tetrahedron2EdgeCrossContainer)
{
    ASSERT_TRUE(this->testInvalidTetrahedron2EdgeContainer());
}

TEST_F(TetrahedronTopologyChecker_test, Check_Invalid_Tetrahedron2VertexCrossContainer)
{
    ASSERT_TRUE(this->testInvalidTetrahedron2VertexContainer());
}



TEST_F(HexahedronTopologyChecker_test, Check_Valid_Topology)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(HexahedronTopologyChecker_test, Check_Invalid_Containers)
{
    ASSERT_TRUE(this->testInvalidContainer());
}

TEST_F(HexahedronTopologyChecker_test, Check_Invalid_Hexahedron2QuadCrossContainer)
{
    ASSERT_TRUE(this->testInvalidHexahedron2QuadContainer());
}

TEST_F(HexahedronTopologyChecker_test, Check_Invalid_Hexahedron2EdgeCrossContainer)
{
    ASSERT_TRUE(this->testInvalidHexahedron2EdgeContainer());
}

TEST_F(HexahedronTopologyChecker_test, Check_Invalid_Hexahedron2VertexCrossContainer)
{
    ASSERT_TRUE(this->testInvalidHexahedron2VertexContainer());
}

}// namespace 
