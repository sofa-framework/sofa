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
#include "stdafx.h"
#include "Sofa_test.h"

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

/** Test a scene: load a given scene with the xml file contained in the sub-directories Scenes and init it.
To test a new scene add the xml file in the Scenes directory and add the TEST_F for your scene (see below the example for BilinearConstraint scene). 
 */
struct LoadScene_test: public Sofa_test<>
{
    // root
   simulation::Node::SPtr root;

   bool LoadScene(std::string sceneName)
   {
       // Init Sofa
       sofa::component::initComponentBase();
       sofa::component::initComponentCommon();
       sofa::component::initComponentGeneral();
       sofa::component::initComponentAdvanced();
       sofa::component::initComponentMisc();

       simulation::Simulation* simulation;
       sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

       // Load the scene from the xml file
       std::string fileName = std::string(SOFASIMULATION_TEST_SCENES_DIR) + "/common/" + sceneName;
       root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));

       // Test if load has succeeded
       sofa::simulation::SceneLoaderXML scene;
       
       if(!root || !scene.loadSucceed)
       {  
           ADD_FAILURE() << "Error while loading the scene: " << sceneName << std::endl;
           return false;   
       }

       return true;
   }

   bool initScene (std::string sceneName)
   {
       LoadScene(sceneName);
      
       // Init the scene
       sofa::simulation::getSimulation()->init(root.get());

       // Test if root is not null
       if(!root)
       {  
           ADD_FAILURE() << "Error in init for the scene: " << sceneName << std::endl;
           return false;   
       }

       return true;

   }
        

};

TEST_F( LoadScene_test,PatchTestConstraint)
{
     ASSERT_TRUE(this->LoadScene("PatchTestConstraint.scn"));
     ASSERT_TRUE(this->initScene("PatchTestConstraint.scn"));
     ASSERT_NO_THROW(this->initScene("PatchTestConstraint.scn"));
}

}// namespace sofa







