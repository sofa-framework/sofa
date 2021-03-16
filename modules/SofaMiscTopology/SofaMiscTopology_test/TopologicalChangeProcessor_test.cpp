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
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

#include <SofaSimulationGraph/SimpleApi.h>
using sofa::helper::testing::BaseSimulationTest;
using namespace sofa::component::topology;

namespace sofa::helper::testing
{

/**  Test TopologicalChangeProcessor incise process
  */

struct TopologicalChangeProcessor_test: public BaseSimulationTest
{
    /// Store SceneInstance 
    BaseSimulationTest::SceneInstance m_instance;

    /// Name of the file to load
    std::string m_fileName = "";

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
            ADD_FAILURE() << "Error while loading the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
            return;
        }
    }

    /// Method to really do the test per type of topology change, to be implemented by child classes
    virtual bool testTopologyChanges() = 0;

    /// Unload the scene
    void TearDown()
    {
        if (m_instance.root !=nullptr)
            sofa::simulation::getSimulation()->unload(m_instance.root);
    }

};


struct InciseProcessor_test : TopologicalChangeProcessor_test
{
    InciseProcessor_test() : TopologicalChangeProcessor_test()
    {
        m_fileName = "IncisionTrianglesProcess.scn";
    }

    bool testTopologyChanges() override
    {
        Node::SPtr root = m_instance.root;

        if (!root)
        {
            ADD_FAILURE() << "Error while loading the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
            return false;
        }


        Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
        if (!nodeTopo)
        {
            ADD_FAILURE() << "Error 'SquareGravity' Node not found in scene: " << "IncisionTrianglesProcess.scn" << std::endl;
            return false;
        }

        TriangleSetTopologyContainer* topoCon = dynamic_cast<TriangleSetTopologyContainer*>(nodeTopo->getMeshTopology());
        if (topoCon == nullptr)
        {
            ADD_FAILURE() << "Error: TriangleSetTopologyContainer not found in 'SquareGravity' Node, in scene: " << "IncisionTrianglesProcess.scn" << std::endl;
            return false;
        }


        // check topology at start
        EXPECT_EQ(topoCon->getNbTriangles(), 1450);
        EXPECT_EQ(topoCon->getNbEdges(), 2223);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        // to test incise animates the scene at least 1.2s
        for (int i = 0; i < 400; i++)
        {
            m_instance.simulate(0.01);
        }

        EXPECT_EQ(topoCon->getNbTriangles(), 1450);
        EXPECT_EQ(topoCon->getNbEdges(), 2223);
        EXPECT_EQ(topoCon->getNbPoints(), 774);

        return true;
    }
};


TEST_F(InciseProcessor_test, Incise)
{
    ASSERT_TRUE(this->testTopologyChanges());
}

}// namespace sofa::helper::testing
