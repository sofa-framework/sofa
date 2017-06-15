/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/Quater.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including component
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {

    using namespace component;
    using namespace defaulttype;

   /// Affine patch test in 3D
  /**  
    * An affine movement is applied to the borders of a mesh. Test if the points within have the same affine movement.
    * This screenshot explains how the patch test works:
    * \image html AffinePatchTest.png 
    * The affine patch test runs with the Assembled Solver of compliant. The table above shows the different test cases achieved:
    * \image html AffinePatchTestCases.png
    */

    template <typename _DataTypes>
    struct AffinePatch_test : public Sofa_test<typename _DataTypes::Real>
    {
        typedef _DataTypes DataTypes;
        typedef typename DataTypes::CPos CPos;
        typedef typename DataTypes::Coord Coord;
        typedef typename DataTypes::VecCoord VecCoord;
        typedef typename DataTypes::VecDeriv VecDeriv;
        typedef typename DataTypes::Real Real;
        typedef projectiveconstraintset::AffineMovementConstraint<DataTypes> AffineMovementConstraint;
        typedef container::MechanicalObject<DataTypes> MechanicalObject;
        typedef defaulttype::Quat Quat;
        typedef defaulttype::Vector3 Vec3;

        /// Root of the scene graph
        simulation::Node::SPtr root;      
        /// Simulation
        simulation::Simulation* simulation; 
        /// Tested Rotation: random rotation matrix
        defaulttype::Mat<3,3,Real> testedRotation;
        /// Tested Translation: random translation
        Vec3 testedTranslation;

        /// Create the context for the scene
        void SetUp()
        { 
            // Init simulation
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

            root = simulation::getSimulation()->createNewGraph("root");
        }

        /// Load the scene to test
        void loadScene(std::string sceneName)
        {
            // Load the scene from the xml file
            std::string fileName = std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + sceneName;
            root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );
        }

        void SetRandomAffineTransform ()
        {
            // Random Matrix 3*3
            for( int j=0; j<testedRotation.nbCols; j++)
            {
                for( int i=0; i<testedRotation.nbLines; i++)
                {
                    testedRotation(i,j)=helper::drand(1);
                }
            }

            // Random Translation
            for(size_t i=0;i<testedTranslation.size();++i)
            {
                testedTranslation[i]=helper::drand(2);
            }

        }
        
        /// After simulation compare the positions of points to the theoretical positions.
        bool compareSimulatedToTheoreticalPositions(double convergenceAccuracy, double diffMaxBetweenSimulatedAndTheoreticalPosition)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(root.get());

            // Compute the theoretical final positions    
            VecCoord finalPos;
            typename AffineMovementConstraint::SPtr affineConstraint  = root->get<AffineMovementConstraint>(root->SearchDown);
            typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(root->SearchDown);
            typename MechanicalObject::ReadVecCoord x0 = dofs->readPositions();
            affineConstraint->getFinalPositions( finalPos,*dofs->write(core::VecCoordId::position()) );
            
            // Set random rotation and translation for affine constraint
            this->SetRandomAffineTransform();

            // Set data values of affine movement constraint
            affineConstraint->m_rotation.setValue(testedRotation);
            affineConstraint->m_translation.setValue(testedTranslation);

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
    TYPED_TEST_CASE(AffinePatch_test, DataTypes);

    // test case: polarcorotationalStrainMapping 
    TYPED_TEST( AffinePatch_test , PolarCorotationalAffinePatchTest)
    {
        // With polar method
        this->loadScene( "PolarCorotationalAffinePatchTest.scn");
        ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-14,1e-13)); 
    }

    // test case: smallcorotationalStrainMapping 
    TYPED_TEST( AffinePatch_test , SmallCorotationalAffinePatchTest)
    {
        // With small method
        this->loadScene( "SmallCorotationalAffinePatchTest.scn");
        ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-15,1e-14)); 
    }

    // test case: svdcorotationalStrainMapping 
    TYPED_TEST( AffinePatch_test , SvdCorotationalAffinePatchTest)
    {
        // With svd method
        this->loadScene( "SvdCorotationalAffinePatchTest.scn");
        ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-14,1e-13));
    }

    // test case: GreenStrainMapping 
    TYPED_TEST( AffinePatch_test , GreenAffinePatchTest)
    {
        this->loadScene( "GreenAffinePatchTest.scn");
        ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-14,1e-13)); 
    }

} // namespace sofa
