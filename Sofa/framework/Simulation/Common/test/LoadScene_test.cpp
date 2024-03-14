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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

namespace sofa {

/** Test a scene: load a given scene with the xml file contained in the sub-directories Scenes and init it.
To test a new scene add the xml file in the Scenes directory and add the TEST_F for your scene (see below the example for BilinearConstraint scene). 
 */
struct LoadScene_test: public BaseTest
{
    // root
   simulation::Node::SPtr root;

   bool LoadScene(std::string sceneName)
   {
       // Load the scene from the xml file
       const std::string fileName = std::string(SOFASIMULATION_TEST_SCENES_DIR) + "/" + sceneName;
       root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::node::load(fileName.c_str()));

       return root != nullptr;
   }

   bool initScene (std::string sceneName)
   {
       LoadScene(sceneName);
      
       // Init the scene
       sofa::simulation::node::initRoot(root.get());

       // Test if root is not null
       if(!root)
       {  
           ADD_FAILURE() << "Error in init for the scene: " << sceneName << std::endl;
           return false;   
       }

       return true;

   }
        

};

TEST_F(LoadScene_test, PatchTestConstraint)
{
    ASSERT_TRUE(this->LoadScene("PatchTestConstraint.scn"));
    ASSERT_TRUE(this->initScene("PatchTestConstraint.scn"));
    ASSERT_NO_THROW(this->initScene("PatchTestConstraint.scn"));
}

TEST_F(LoadScene_test, PythonExtension)
{
    EXPECT_MSG_EMIT(Error);
    ASSERT_FALSE(this->LoadScene("fakeFile.py"));
    ASSERT_FALSE(this->LoadScene("fakeFile.pyscn"));
}

}// namespace sofa







