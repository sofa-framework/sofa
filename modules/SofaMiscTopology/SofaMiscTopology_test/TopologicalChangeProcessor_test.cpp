/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>

namespace sofa {

/**  Test TopologicalChangeProcessor incise process
  */

struct TopologicalChangeProcessor_test: public Sofa_test<>
{
    // root
   simulation::Node::SPtr root;
   /// Simulation
   simulation::Simulation* simulation;

   void SetUp()
   {
       // Init Sofa
       sofa::component::initComponentBase();
       sofa::component::initComponentCommon();
       sofa::component::initComponentGeneral();
       sofa::component::initComponentAdvanced();
       sofa::component::initComponentMisc();

       sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
       root = simulation::getSimulation()->createNewGraph("root");

       // Load the scene from the xml file
       std::string fileName = std::string(SOFAMISCTOPOLOGY_TEST_SCENES_DIR) + "/" + "IncisionTrianglesProcess.scn";
       std::cout << fileName.c_str() << std::endl;
       root = sofa::simulation::getSimulation()->load(fileName.c_str()).get();

       // Test if root is not null
       if(!root)
       {
           ADD_FAILURE() << "Error while loading the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
           return;
       }

       // Init scene
       sofa::simulation::getSimulation()->init(root.get());

       // Test if root is not null
       if(!root)
       {
           ADD_FAILURE() << "Error in init for the scene: " << "IncisionTrianglesProcess.scn" << std::endl;
           return;
       }
   }

   bool TestInciseProcess()
   {
       // To test incise animates the scene at least 1.2s
       for(int i=0;i<50;i++)
       {
          sofa::simulation::getSimulation()->animate(root.get(),0.1);
       }

       return true;
   }

   /// Unload the scene
   void TearDown()
   {
       if (root!=NULL)
           sofa::simulation::getSimulation()->unload(root);
   }

};

TEST_F( TopologicalChangeProcessor_test,Incise)
{
    ASSERT_TRUE(this->TestInciseProcess());
}

}// sofa
