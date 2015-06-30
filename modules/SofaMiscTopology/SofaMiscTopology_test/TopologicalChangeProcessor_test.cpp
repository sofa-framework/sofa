#include <SofaTest/Sofa_test.h>
#include <SofaComponentMain/init.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

namespace sofa {

/**  Test TopologicalChangeProcessor incise process
  */

struct TopologicalChangeProcessor_test: public Sofa_test<double>
{
    // root
   simulation::Node::SPtr root;

   void SetUp()
   {
       // Init Sofa
       sofa::component::init();
       simulation::Simulation* simulation;
       sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

       // Load the scene from the xml file
       std::string fileName = std::string(SOFAMISCTOPOLOGY_TEST_SCENES_DIR) + "/" + "IncisionTrianglesProcess.scn";
       root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));

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
       // Animate during 3 s
       for(int i=0;i<30;i++)
       {
          sofa::simulation::getSimulation()->animate(root.get(),0.1);
       }

       return true;
   }

};

TEST_F( TopologicalChangeProcessor_test,Incise)
{
    ASSERT_TRUE(this->TestInciseProcess());
}

}// sofa
