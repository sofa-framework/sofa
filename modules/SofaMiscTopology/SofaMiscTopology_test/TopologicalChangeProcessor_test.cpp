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
    // root
   Node::SPtr root;
   /// Simulation
   simulation::Simulation* simulation;

   /// Store SceneInstance 
   BaseSimulationTest::SceneInstance m_instance;
   bool fredDebugMe = false;

   void SetUp()
   {
       sofa::simpleapi::importPlugin("SofaComponentAll");
       // Load the scene from the xml file
       std::string fileName = std::string(SOFAMISCTOPOLOGY_TEST_SCENES_DIR) + "/" + "IncisionTrianglesProcess.scn";
       std::cout << fileName.c_str() << std::endl;

       //m_instance = BaseSimulationTest::SceneInstance::LoadFromFile(fileName);
       m_instance = BaseSimulationTest::SceneInstance();
       m_instance.loadSceneFile(fileName);
       root = m_instance.root;
       
       if (fredDebugMe)
       {
           std::cout << "root: " << root.get() << std::endl;
           std::cout << "m_instance.root: " << m_instance.root << std::endl;
           std::cout << root.get()->getName() << std::endl;
           std::cout << m_instance.root.get()->getChildren().size() << std::endl;
       }
       // Init scene
       m_instance.initScene();

       root = m_instance.root;

       if (fredDebugMe)
       {
           std::cout << root.get()->getName() << std::endl;
           std::cout << m_instance.root.get()->getChildren().size() << std::endl;
       }

       // Test if root is not null
       if(!root)
       {
           ADD_FAILURE() << "Error while loading the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
           return;
       }
     
   }

   bool TestInciseProcess()
   {
       Node::SPtr root = m_instance.root;

       if (fredDebugMe)
       {
           std::cout << "root: " << root.get() << std::endl;
           std::cout << root.get()->getName() << std::endl;
           std::cout << m_instance.root.get()->getChildren().size() << std::endl;
       }

       if (!root)
       {
           ADD_FAILURE() << "Error while loading the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
           return false;
       }


     /*  Node::SPtr nodeTopo = root.get()->getChild("SquareGravity");
       if (!nodeTopo)
       {
           ADD_FAILURE() << "Error 'SquareGravity' Node not found in scene: " << "IncisionTrianglesProcess.scn" << std::endl;
           return false;
       }*/


       // to test incise animates the scene at least 1.2s
       for(int i=0;i<50;i++)
       {
           m_instance.simulate(0.1);
       }

       return true;
   }

   /// Unload the scene
   void TearDown()
   {
       if (m_instance.root !=nullptr)
           sofa::simulation::getSimulation()->unload(m_instance.root);
   }

};

TEST_F( TopologicalChangeProcessor_test,Incise)
{
    ASSERT_TRUE(this->TestInciseProcess());
}

}// namespace sofa::helper::testing
