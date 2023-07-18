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
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic//TetrahedronSetTopologyContainer.h>

#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/simulation/Node.h>

using sofa::testing::BaseSimulationTest;

namespace 
{

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::simulation;

/**  Test TopologicalChangeProcessor incise process
  */

struct TopologicalChangeProcessor_test: public BaseSimulationTest
{
    /// Store SceneInstance 
    BaseSimulationTest::SceneInstance m_instance;

    /// Name of the file to load
    std::string m_fileName = "";

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

    /// Method to really do the test per type of topology change, to be implemented by child classes
    virtual bool testTopologyChanges() = 0;

    /// Unload the scene
    void TearDown() override
    {
        if (m_instance.root !=nullptr)
            sofa::simulation::node::unload(m_instance.root);
    }

};


struct InciseProcessor_test : TopologicalChangeProcessor_test
{
    InciseProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "/IncisionTrianglesProcess.scn";
    }

    bool testTopologyChanges() override
    {
        const Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        const Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
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

        // to test incise animates the scene at least 1.2s
        for (int i = 0; i < 50; i++)
        {
            m_instance.simulate(0.05);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 1680);
        EXPECT_EQ(topoCon->getNbEdges(), 2710);
        EXPECT_EQ(topoCon->getNbPoints(), 1029);

        return true;
    }
};


struct RemoveTriangleProcessor_test : TopologicalChangeProcessor_test
{
    RemoveTriangleProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "/RemovingTrianglesProcess.scn";
    }

    bool testTopologyChanges() override
    {
        const Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        const Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
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
};


struct AddTriangleProcessor_test : TopologicalChangeProcessor_test
{
    AddTriangleProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "/AddingTrianglesProcess.scn";
    }

    bool testTopologyChanges() override
    {
        const Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        const Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
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
        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 0);
        EXPECT_EQ(topoCon->getNbPoints(), 27);

        // to test incise animates the scene at least 1.2s
        for (int i = 0; i < 100; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 24);
        EXPECT_EQ(topoCon->getNbEdges(), 42);
        EXPECT_EQ(topoCon->getNbPoints(), 27);

        return true;
    }
};



struct RemoveTetrahedronProcessor_test : TopologicalChangeProcessor_test
{
    RemoveTetrahedronProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "/RemovingTetraProcess.scn";
    }

    bool testTopologyChanges() override
    {
        const Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        const Node::SPtr nodeTopo = root.get()->getChild("TT");
        if (!nodeTopo)
        {
            ADD_FAILURE() << "Error 'TT' Node not found in scene: " << m_fileName << std::endl;
            return false;
        }

        TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(nodeTopo->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TetrahedronSetTopologyContainer not found in 'TT' Node, in scene: " << m_fileName << std::endl;
            return false;
        }


        // check topology at start
        EXPECT_EQ(topoCon->getNbTetrahedra(), 44);
        EXPECT_EQ(topoCon->getNbTriangles(), 112);
        EXPECT_EQ(topoCon->getNbEdges(), 93);
        EXPECT_EQ(topoCon->getNbPoints(), 26);


        for (int i = 0; i < 20; i++)
        {
            m_instance.simulate(0.01);
        }


        EXPECT_EQ(topoCon->getNbTetrahedra(), 34);
        EXPECT_EQ(topoCon->getNbTriangles(), 97);
        EXPECT_EQ(topoCon->getNbEdges(), 86);
        EXPECT_EQ(topoCon->getNbPoints(), 25);

        return true;
    }
};


struct AddTetrahedronProcessor_test : TopologicalChangeProcessor_test
{
    AddTetrahedronProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "/AddingTetraProcess.scn";
    }

    bool testTopologyChanges() override
    {
        const Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << m_fileName << std::endl;
            return false;
        }


        const Node::SPtr nodeTopo = root.get()->getChild("TT");
        if (!nodeTopo)
        {
            ADD_FAILURE() << "Error 'TT' Node not found in scene: " << m_fileName << std::endl;
            return false;
        }

        TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(nodeTopo->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TetrahedronSetTopologyContainer not found in 'TT' Node, in scene: " << m_fileName << std::endl;
            return false;
        }


        // check topology at start
        EXPECT_EQ(topoCon->getNbTetrahedra(), 0);
        EXPECT_EQ(topoCon->getNbTriangles(), 0);
        EXPECT_EQ(topoCon->getNbEdges(), 0);
        EXPECT_EQ(topoCon->getNbPoints(), 27);

        for (int i = 0; i < 41; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTetrahedra(), 8);
        EXPECT_EQ(topoCon->getNbTriangles(), 24);
        EXPECT_EQ(topoCon->getNbEdges(), 25);
        EXPECT_EQ(topoCon->getNbPoints(), 27);

        return true;
    }
};


TEST_F(RemoveTetrahedronProcessor_test, RemoveTetrahedra)
{
    ASSERT_TRUE(this->testTopologyChanges());
}

TEST_F(AddTetrahedronProcessor_test, AddTetrahedra)
{
    ASSERT_TRUE(this->testTopologyChanges());
}



TEST_F(InciseProcessor_test, InciseTriangles)
{
    ASSERT_TRUE(this->testTopologyChanges());
}

TEST_F(AddTriangleProcessor_test, AddTriangles)
{
    ASSERT_TRUE(this->testTopologyChanges());
}

TEST_F(RemoveTriangleProcessor_test, RemoveTriangles)
{
    ASSERT_TRUE(this->testTopologyChanges());
}




}// namespace 
