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
#include <sofa/helper/Quater.h>

// Including component
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include "../deformationMapping/LinearMapping.h"

#include <SofaTest/Mapping_test.h>

namespace sofa {

    using namespace defaulttype;
    using namespace component;
    using namespace component::mapping;


   /// Linear Deformation mappings test
   /**  
    * Test the deformation mappings by applying a rigid transformation (translation and rotation) 
    * to all rigid nodes and checks if the resulting deformation gradient is equal to rotation.
    * This screenshot explains how this test works:
    * \image html RigidDeformationMapping.png
   */

    template <typename _Mapping>
    struct RigidLinearDeformationMappings_test : public Mapping_test<_Mapping>
    {
        typedef Mapping_test<_Mapping> Inherited;
        typedef typename Inherited::In In;
        typedef typename In::CPos CPos;
        typedef typename In::Coord Coord;
        typedef typename Inherited::Out Out;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord; 
        typedef typename Out::Frame OutFrame;
        typedef component::container::MechanicalObject<In> InDOFs;
        typedef component::container::MechanicalObject<Out> OutDOFs;
        typedef defaulttype::Quat Quat;
        typedef defaulttype::Vector3 Vec3;

        /// Tested Rotation: random rotation matrix  
        defaulttype::Mat<3,3,Real> testedRotation; 
        Quat testedQuaternion;
        /// Tested Translation: random translation
        Vec3 testedTranslation;

        // Constructor: call the constructor of the base class which loads the scene to test
        RigidLinearDeformationMappings_test() : Mapping_test<_Mapping>(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "RigidLineDeformationMapping.scn")
        {
            // rotation and translation
            this->SetRandomTestedRotationAndTranslation();
            typedef projectiveconstraintset::AffineMovementConstraint<In> AffineMovementConstraint;
            typename AffineMovementConstraint::SPtr affineConstraint  = this->root->template get<AffineMovementConstraint>(this->root->SearchDown);
            affineConstraint->m_quaternion.setValue(testedQuaternion);
            affineConstraint->m_translation.setValue(testedTranslation);
            testedQuaternion.toMatrix(testedRotation);

            static_cast<_Mapping*>(this->mapping)->d_geometricStiffness.setValue( 1 );
        }
             
        void SetRandomTestedRotationAndTranslation()
        {
            // Random Rotation
            SReal x,y,z,w;
            // Random axis
            x = SReal(helper::drand(1));
            y = SReal(helper::drand(1));
            z = SReal(helper::drand(1));
            // If the rotation axis is null
            Vec3 rotationAxis(x,y,z);
            if(rotationAxis.norm() < 1e-7)
            {
                rotationAxis = Vec3(0,0,1);
            }
            rotationAxis.normalize();
            // Random angle between 0 and M_PI
            w = helper::drand()* M_PI;
            // Quat = (rotationAxis*sin(angle/2) , cos(angle/2)) angle = 2*w
            testedQuaternion = Quat(sin(w)*rotationAxis[0],rotationAxis[1]*sin(w),rotationAxis[2]*sin(w),cos(w));
   
            // Translation
           for(size_t i=0;i<testedTranslation.size();++i)
           {
               testedTranslation[i]=helper::drand(2);
           }

        }

        using Inherited::runTest;
        /// After simulation compare the positions of deformation gradients to the theoretical positions.
        bool runTest(double convergenceAccuracy)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(this->root.get());

            // Get dofs positions
            typename  InDOFs::ReadVecCoord x = this->inDofs->readPositions();
            
            // xin 
            InVecCoord xin(x.size());
            copyFromData(xin,x);
    
            // xout
            typename  OutDOFs::ReadVecCoord xelasticityDofs = this->outDofs->readPositions();
            OutVecCoord xout(xelasticityDofs.size());
            copyFromData(xout,xelasticityDofs);

            // Initialize parameters to test convergence
            size_t numNodes = xin.size();
            InVecCoord xPrev(numNodes);
            InVecCoord dx(numNodes); 
            bool hasConverged = true;

            for (size_t i=0; i<numNodes; i++)
            {
                xPrev[i] = CPos(0,0,0);
            }

            // Animate
            do
            { 
                hasConverged = true;
                sofa::simulation::getSimulation()->animate(this->root.get(),0.05);
                typename InDOFs::ReadVecCoord xCurrent = this->inDofs->readPositions();

                // Compute dx
                for (size_t i=0; i<xCurrent.size(); i++)
                {
                    // Translation
                    dx[i].getCenter() = xCurrent[i].getCenter()-xPrev[i].getCenter();
                    //Rotation
                    dx[i].getOrientation() = xCurrent[i].getOrientation().inverse()*xPrev[i].getOrientation();
                    // Test convergence
                    if(dx[i].norm()>convergenceAccuracy) hasConverged = false;
                }

                // xprev = xCurrent
                for (size_t i=0; i<numNodes; i++)
                {
                    xPrev[i]=xCurrent[i];
                }
            }
            while(!hasConverged); // not converged

            // Parent new : Get simulated positions
            typename InDOFs::WriteVecCoord xinNew = this->inDofs->writePositions();
     
            // New position of parents
            InVecCoord parentNew(xinNew.size());
            for(size_t i=0; i<xinNew.size();++i)
            {
                parentNew[i] = xinNew[i];
            }
   
            // Expected children positions: rotation from affine constraint
            typename OutDOFs::WriteVecCoord xoutNew = this->outDofs->writePositions();
            OutVecCoord expectedChildCoords(xoutNew.size());
  
            for(size_t i=0;i<xoutNew.size();++i)
            {
                OutFrame &f = expectedChildCoords[i].getF();
                f = testedRotation;
            }

           // run the mapping test
           return Inherited::runTest(xin,xout,parentNew,expectedChildCoords);

        }

    };

      // Define the list of DataTypes to instantiate
    using testing::Types;
    typedef Types<
        LinearMapping<Rigid3Types, F331Types>,
        LinearMapping<Rigid3Types, F332Types>
    > DataTypes; // the types to instantiate.

    // Test suite for all the instantiations
    TYPED_TEST_CASE(RigidLinearDeformationMappings_test, DataTypes);

    // test case: polarcorotationalStrainMapping 
    TYPED_TEST( RigidLinearDeformationMappings_test , RigidStrainDeformationPatchTest)
    {
        ASSERT_TRUE( this->runTest(1e-15));
    }

} // namespace sofa
