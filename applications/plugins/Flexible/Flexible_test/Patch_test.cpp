/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "stdafx.h"
#include <SofaTest/Sofa_test.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including component
#include <SofaBoundaryCondition/PatchTestMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {

    using namespace component;
    using namespace defaulttype;

    /// Patch test 

    /**  Patch test in 3D with Flexible.
    * The tested scene is contained in the scenes subdirectory. The test works with the assembled solver of compliant and the small method of corotational strain mapping.
    * A movement is applied to the borders of a mesh. The points within should have a linear movement relative to the border movements.
    * This screenshot explains how the patch test works:
    *  \image html PatchTest.png
   */

    template <typename _DataTypes>
    struct Patch_test : public Sofa_test<typename _DataTypes::Real>
    {
        typedef _DataTypes DataTypes;
        typedef typename DataTypes::CPos CPos;
        typedef typename DataTypes::VecCoord VecCoord;
        typedef typename DataTypes::VecDeriv VecDeriv;
        typedef projectiveconstraintset::PatchTestMovementConstraint<DataTypes> PatchTestMovementConstraint;
        typedef container::MechanicalObject<DataTypes> MechanicalObject;

        /// Root of the scene graph
        simulation::Node::SPtr root;      
        /// Simulation
        simulation::Simulation* simulation;  

        /// Create the context for the scene
        void SetUp()
        { 
            // Init simulation
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

            root = simulation::getSimulation()->createNewGraph("root");
        }

        /// Load the scene to test SmallCorotationalPatchTest.scn
        void loadScene(std::string sceneName)
        {
            // Load the scene from the xml file
            std::string fileName = std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + sceneName;
            root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );
        }

        /// After simulation compare the positions of points to the theoretical positions.
        bool compareSimulatedToTheoreticalPositions(double convergenceAccuracy, double diffMaxBetweenSimulatedAndTheoreticalPosition)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(root.get());

            // Compute the theoretical final positions    
            VecCoord finalPos;
            typename PatchTestMovementConstraint::SPtr bilinearConstraint  = root->get<PatchTestMovementConstraint>(root->SearchDown);
            typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(root->SearchDown);
            typename MechanicalObject::ReadVecCoord x0 = dofs->readPositions();
            bilinearConstraint->getFinalPositions( finalPos,*dofs->write(core::VecCoordId::position()) );

            // Initialize
            size_t numNodes = finalPos.size();
            VecCoord xprev(numNodes);
            VecDeriv dx(numNodes); 
            bool hasConverged = true;
            
            for (size_t i=0; i<numNodes; i++)
            {
                xprev[i] = CPos(0,0,0);
            }

            // Animate
            do
            {
                hasConverged = true;
                sofa::simulation::getSimulation()->animate(root.get(),0.5);
                typename MechanicalObject::ReadVecCoord x = dofs->readPositions();

                // Compute dx
                for (size_t i=0; i<x.size(); i++)
                {
                    dx[i] = x[i]-xprev[i];
                    // Test convergence
                    if(dx[i].norm()>convergenceAccuracy) hasConverged = false;
                }
               
                // xprev = x
                for (size_t i=0; i<numNodes; i++)
                {
                    xprev[i]=x[i];
                }
            }
            while(!hasConverged); // not converged

            // Compare the theoretical positions and the simulated positions   
            bool succeed=true;
            for(size_t i=0; i<finalPos.size(); i++ )
            {
                if((finalPos[i]-x0[i]).norm()>diffMaxBetweenSimulatedAndTheoreticalPosition)
                {   
                    succeed = false;
                    ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x0[i] << std::endl <<"the expected Position is " << finalPos[i] << std::endl
                        << "difference = " <<(finalPos[i]-x0[i]).norm() << std::endl;
                }
            }
            return succeed;
        }

        /// Unload the scene
        void TearDown()
        {
            if (root!=NULL)
                sofa::simulation::getSimulation()->unload(root);
        }

    };

    // Define the list of DataTypes to instantiate
    using testing::Types;
    typedef Types<
        Vec3Types
    > DataTypes; // the types to instantiate.

    // Test suite for all the instantiations
    TYPED_TEST_CASE(Patch_test, DataTypes);

    // test case: smallcorotationalStrainMapping 
    TYPED_TEST( Patch_test , SmallCorotationalPatchTest)
    {
        // With small method
        this->loadScene( "SmallCorotationalPatchTest.scn");
        ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-12,3e-2));
    }

} // namespace sofa
