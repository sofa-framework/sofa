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
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaMiscTopology/TopologyChecker.h>

#include <SofaSimulationGraph/SimpleApi.h>
#include <sofa/simulation/Node.h>

using sofa::helper::testing::BaseSimulationTest;

namespace 
{

using namespace sofa::component::topology;
using namespace sofa::core::topology;
using namespace sofa::simulation;
using sofa::component::misc::TopologyChecker;

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
    void SetUp()
    {
        sofa::simpleapi::importPlugin("SofaComponentAll");
        // Load the scene from the xml file
        std::string filePath = std::string(SOFAMISCTOPOLOGY_TEST_SCENES_DIR) + "/" + m_fileName;
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
    virtual bool testInvalidTopology() = 0;


    /// Unload the scene
    void TearDown()
    {
        if (m_instance.root !=nullptr)
            sofa::simulation::getSimulation()->unload(m_instance.root);
    }

};


struct EdgeTopologyChecker_test : TopologyChecker_test
{
    EdgeTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "TopologicalModifiers/RemovingTriangle2EdgeProcess.scn";
    }


    EdgeSetTopologyContainer* loadTopo()
    {
        Node::SPtr root = m_instance.root;
        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return nullptr;
        }


        Node::SPtr nodeTopoTri = root.get()->getChild("SquareGravity");
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
        EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
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
        auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                {"name", "checker"},
                                {"topology", topoLink}
            });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        // check topology at start
        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 96);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        EXPECT_EQ(checker->checkContainer(), true);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 333);
        EXPECT_EQ(topoCon->getNbPoints(), 261);

        return checker->checkContainer() == true;
    }


    bool testInvalidTopology() override
    {
        EdgeSetTopologyContainer* topoCon = loadTopo();
        if (topoCon == nullptr)
            return false;

        std::string topoLink = "@" + topoCon->getName();
        auto obj = sofa::simpleapi::createObject(m_topoNode, "TopologyChecker", {
                                {"name", "checker"},
                                {"topology", topoLink}
            });
        TopologyChecker* checker = dynamic_cast<TopologyChecker*>(obj.get());
        checker->init();

        sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<sofa::helper::vector<Topology::Edge> > > edges = topoCon->d_edge;
        
        // Add edges without updating cross container
        edges.push_back(Topology::Edge(0, 10));
        EXPECT_EQ(topoCon->getNbEdges(), 97);

        // Topology checher should detect an error
        EXPECT_MSG_EMIT(Error);

        return checker->checkContainer() == false;
    }
};


struct TriangleTopologyChecker_test : TopologyChecker_test
{
    TriangleTopologyChecker_test() : TopologyChecker_test()
    {
        m_fileName = "RemovingTrianglesProcess.scn";
    }

    bool testValidTopology() override
    {
        Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
        if (!nodeTopo)
        {
            ADD_FAILURE() << "Error 'SquareGravity' Node not found in scene: " << m_fileName << std::endl;
            return false;
        }

        TriangleSetTopologyContainer* topoCon = dynamic_cast<TriangleSetTopologyContainer*>(nodeTopo->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TriangleSetTopologyContainer not found in 'SquareGravity' Node, in scene: " << m_fileName << std::endl;
            return false;
        }


        // check topology at start
        EXPECT_EQ(topoCon->getNbTriangles(), 1450);
        EXPECT_EQ(topoCon->getNbEdges(), 2223);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        //// to test incise animates the scene at least 1.2s
        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 145);
        EXPECT_EQ(topoCon->getNbEdges(), 384);
        EXPECT_EQ(topoCon->getNbPoints(), 261);

        return true;
    }

    bool testInvalidTopology() override
    {
        return true;
    }
};



TEST_F(EdgeTopologyChecker_test, UnsetTopologyLink)
{
    ASSERT_TRUE(this->testUnsetTopologyLink());
}

TEST_F(EdgeTopologyChecker_test, Check_Valid_EdgeContainers)
{
    ASSERT_TRUE(this->testValidTopology());
}

TEST_F(EdgeTopologyChecker_test, Check_Invalid_EdgeContainers)
{
    ASSERT_TRUE(this->testInvalidTopology());
}




}// namespace 
