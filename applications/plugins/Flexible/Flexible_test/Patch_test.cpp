
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <plugins/SofaTest/Sofa_test.h>
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <ComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including component
#include <BoundaryCondition/projectiveconstraintset/BilinearMovementConstraint.h>
#include <BaseMechanics/MechanicalObject.h>


namespace sofa {

    using namespace component;
    using namespace defaulttype;

    /**  Patch test in 3D with Flexible. The tested scene is contained in share/tests/Flexible/PatchTest.scn.
    A movement is applied to the borders of a mesh. The points within should have a bilinear movement relative to the border movements.*/

    template <typename _DataTypes>
    struct Patch_test : public Sofa_test<typename _DataTypes::Real>
    {
        typedef _DataTypes DataTypes;
        typedef typename DataTypes::VecCoord VecCoord;
        typedef projectiveconstraintset::BilinearMovementConstraint<DataTypes> BilinearMovementConstraint;
        typedef container::MechanicalObject<DataTypes> MechanicalObject;

        /// Root of the scene graph
        simulation::Node::SPtr root;      
        /// Simulation
        simulation::Simulation* simulation;  

        // Create the context for the scene
        void SetUp()
        { 
            // Init simulation
            sofa::component::init();
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

            root = simulation::getSimulation()->createNewGraph("root");
        }

        void loadScene(std::string sceneName)
        {
            // Load the scene from the xml file
            std::string fileName = sofa::helper::system::DataRepository.getFile(sceneName);
            root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));
        }

        bool test_projectPosition()
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(root.get());

            // Animate
            do
            {sofa::simulation::getSimulation()->animate(root.get(),0.5);}
            while(root->getAnimationLoop()->getTime() < 40); 

            // Get the simulated final positions
            typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(root->SearchDown);
            typename MechanicalObject::WriteVecCoord x = dofs->writePositions();

            // Compute the theoretical final positions    
            typename BilinearMovementConstraint::SPtr bilinearConstraint = root->get<BilinearMovementConstraint>(root->SearchDown);
            VecCoord finalPos;
            bilinearConstraint->getFinalPositions( finalPos,*dofs->write(core::VecCoordId::position()) );

            // Compare the theoretical positions and the simulated positions   
            bool succeed=true;
            for(size_t i=0; i<finalPos.size(); i++ )
            {
                if((finalPos[i]-x[i]).norm()>2.3e-4)
                {   
                    succeed = false;
                    ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x[i] << std::endl <<"the expected Position is " << finalPos[i] << std::endl 
                        << "difference = " <<(finalPos[i]-x[i]).norm() << std::endl;
                }
            }
            return succeed;
        }


        void TearDown()
        {
            if (root!=NULL)
                sofa::simulation::getSimulation()->unload(root);
            //        cerr<<"tearing down"<<endl;
        }

    };

    // Define the list of DataTypes to instantiate
    using testing::Types;
    typedef Types<
        Vec3Types
    > DataTypes; // the types to instantiate.

    // Test suite for all the instantiations
    TYPED_TEST_CASE(Patch_test, DataTypes);

    // test case
    TYPED_TEST( Patch_test , patchTest3D )
    {
        this->loadScene( "tests/Flexible/PatchTest.scn");
        ASSERT_TRUE( this->test_projectPosition());
    }

} // namespace sofa
