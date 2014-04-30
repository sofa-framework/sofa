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
#include <sofa/component/init.h>
#include <sofa/core/ExecParams.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including component
#include <sofa/component/projectiveconstraintset/AffineMovementConstraint.h>
#include <sofa/component/container/MechanicalObject.h>
#include "../deformationMapping/LinearMapping.h"
#include "../shapeFunction/BarycentricShapeFunction.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../quadrature/TopologyGaussPointSampler.h"

#include <plugins/SceneCreator/SceneCreator.h>

#include <Mapping_test.h>

namespace sofa {

    using namespace defaulttype;
    using namespace component;
    using namespace component::mapping;


   /// Linear Deformation mappings test
   /**  
    * Test the deformation mappings by applying an affine transformation (translation and rotation) 
    * to all control nodes and checks if the resulting deformation mapping is equal to rotation.
   */

    template <typename _Mapping>
    struct RigidLinearDeformationMappings_test : public Mapping_test<_Mapping>
    {
        typedef Mapping_test<_Mapping> Inherited;
        typedef typename Inherited::In In;
        typedef typename In::CPos CPos;
        typedef typename In::Coord Coord;
        typedef typename Inherited::Out Out;
        typedef typename Inherited::ReadInVecCoord ReadInVecCoord;
        typedef typename Inherited::ReadOutVecCoord ReadOutVecCoord;
        typedef typename Inherited::WriteInVecCoord WriteInVecCoord;
        typedef typename Inherited::WriteOutVecCoord WriteOutVecCoord;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutCoord OutCoord;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename Inherited::InVecDeriv InVecDeriv;
        typedef typename Out::Frame OutFrame;
        typedef component::container::MechanicalObject<In> InDOFs;
        typedef component::container::MechanicalObject<Out> OutDOFs;
        typedef defaulttype::Quat Quat;
        typedef defaulttype::Vector3 Vec3;

        /// Tested Rotation: random rotation matrix  
        defaulttype::Mat<3,3,Real> testedRotation; 
        Quat testedQuaternion;
        /// Tested Translation: random translation
        Coord testedTranslation;
        /// Seed for random value
        long seed;
        /// Random generator
        sofa::helper::RandomGenerator randomGenerator;

        // Constructor: call the constructor of the base class which loads the scene to test
        RigidLinearDeformationMappings_test() : Mapping_test(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "RigidLineDeformationMapping.scn")
        {   
            Inherited::errorMax = 5000;
         
            seed=2;
            // rotation and translation
            randomGenerator.initSeed(seed);
            this->SetRandomTestedRotationAndTranslation(seed);
            typedef projectiveconstraintset::AffineMovementConstraint<In> AffineMovementConstraint;
            AffineMovementConstraint::SPtr affineConstraint  = root->get<AffineMovementConstraint>(root->SearchDown);
            affineConstraint->m_quaternion.setValue(testedQuaternion);
            affineConstraint->m_translation.setValue(testedTranslation);
            testedQuaternion.toMatrix(testedRotation);

        }
             
        void SetRandomTestedRotationAndTranslation(int seedValue)
        {
            // Random Rotation
            SReal x,y,z,w;
            // Random axis
            x = randomGenerator.random<SReal>(-1.0,1.0);
            y = randomGenerator.random<SReal>(-1.0,1.0);
            z = randomGenerator.random<SReal>(-1.0,1.0);   
            // If the rotation axis is null
            Vec3 rotationAxis(x,y,z);
            if(rotationAxis.norm() < 1e-7)
            {
                rotationAxis = Vec3(0,0,1);
            }
            rotationAxis.normalize();
            // Random angle
            w = randomGenerator.random<SReal>(0.0, M_PI);
            // Quat = (rotationAxis*sin(angle/2) , cos(angle/2)) angle = 2*w
            testedQuaternion = Quat(sin(w)*rotationAxis[0],rotationAxis[1]*sin(w),rotationAxis[2]*sin(w),cos(w));
   
            // Translation
           for(size_t i=0;i<Coord::total_size;++i)
           {
               if(i<3)
               testedTranslation[i]=randomGenerator.random<SReal>(-2.0,2.0);
               else
               testedTranslation[i]=testedQuaternion[i]; 
           }

        }

        /// After simulation compare the positions of deformation gradients to the theoretical positions.
        bool runTest(double convergenceAccuracy)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(root.get());

            // Get dofs positions
            typename  InDOFs::ReadVecCoord x = inDofs->readPositions();
            
            // xin 
            InVecCoord xin(x.size());
            copyFromData(xin,x);
    
            // xout
            typename  OutDOFs::ReadVecCoord xelasticityDofs = outDofs->readPositions();
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
                sofa::simulation::getSimulation()->animate(root.get(),0.05);
                typename InDOFs::ReadVecCoord xCurrent = inDofs->readPositions();

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
            typename InDOFs::WriteVecCoord xinNew = inDofs->writePositions();
     
            // New position of parents
            InVecCoord parentNew(xinNew.size());
            for(size_t i=0; i<xinNew.size();++i)
            {
                parentNew[i] = xinNew[i];
            }
   
            // Expected children positions: rotation from affine constraint
            typename OutDOFs::WriteVecCoord xoutNew = outDofs->writePositions();
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
    TYPED_TEST( RigidLinearDeformationMappings_test , StrainDeformationPatchTest)
    {
        ASSERT_TRUE( this->runTest(1e-15));
    }

} // namespace sofa