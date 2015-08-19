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
   /// Simulation
   simulation::Simulation* simulation;

   void SetUp()
   {
       // Init Sofa
       sofa::component::init();
       sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
       root = simulation::getSimulation()->createNewGraph("root");

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
