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
#include "stdafx.h"
#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

// Including component

#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include "../deformationMapping/LinearMapping.h"

#include <Mapping_test.h>

namespace sofa {

    using namespace defaulttype;
    using namespace component;
    using namespace component::mapping;


   /// Linear Deformation mappings test
  /**  
    * Test the deformation mappings by applying an affine transformation 
    * to all dofs and checks if the resulting deformation gradient is equal to the transformation.
    * This screenshot explains how this test works:
    * \image html PointDeformationMapping.png
    */

    template <typename _Mapping>
    struct PointsDeformationMapping_test : public Mapping_test<_Mapping>
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
        typedef projectiveconstraintset::AffineMovementConstraint<In> InAffineMovementConstraint;

        /// Tested Rotation: random rotation matrix  
        defaulttype::Mat<3,3,Real> testedRotation;
        /// Tested Translation: random translation
        Vec<3,Real> testedTranslation;
        /// Random generator
        sofa::helper::RandomGenerator randomGenerator;

        // Constructor: call the constructor of the base class which loads the scene to test
        PointsDeformationMapping_test() : Inherited::Mapping_test(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "LinearDeformationMappingPoint.scn")
        {   
            // rotation and translation
            randomGenerator.initSeed(BaseSofa_test::seed);
            this->SetRandomAffineTransform();

            // Get rotation from affine constraint
            simulation::Node::SPtr rootNode =  Inherited::root;
            typename InAffineMovementConstraint::SPtr affineConstraint = rootNode->get<InAffineMovementConstraint>(rootNode->SearchDown);
            affineConstraint->m_rotation.setValue(testedRotation);
            affineConstraint->m_translation.setValue(testedTranslation);
        }
             
        void SetRandomAffineTransform ()
        {
            // Matrix 3*3
            for( int j=0; j<testedRotation.nbCols; j++)
            {
                for( int i=0; i<testedRotation.nbLines; i++)
                {
                    Real random = randomGenerator.random<Real>( (Real) -1, (Real) 1 );
                    testedRotation(i,j)=random;
                }
            }

            // Translation
            for(size_t i=0;i<testedTranslation.size();++i)
            {
                testedTranslation[i]=randomGenerator.random<SReal>(-2.0,2.0);
            }

        }
      
        bool runTest(double convergenceAccuracy)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(Inherited::root.get());

            // Get dofs positions
            ReadInVecCoord x = Inherited::inDofs->readPositions();
            
            // xin 
            InVecCoord parentInit(x.size());
            copyFromData(parentInit,x);
    
            // xout
            ReadOutVecCoord xelasticityDofs = Inherited::outDofs->readPositions();
            OutVecCoord childInit(xelasticityDofs.size());
            copyFromData(childInit,xelasticityDofs);

            // Initialize parameters to test convergence
            size_t numNodes = parentInit.size();
            InVecCoord xPrev(numNodes);
            InVecDeriv dx(numNodes); 
            bool hasConverged = true;

            for (size_t i=0; i<numNodes; i++)
            {
                xPrev[i] = CPos(0,0,0);
            }

            // Animate
            do
            {
                hasConverged = true;
                sofa::simulation::getSimulation()->animate(Inherited::root.get(),0.05);
                ReadInVecCoord xCurrent = Inherited::inDofs->readPositions();

                // Compute dx
                for (size_t i=0; i<xCurrent.size(); i++)
                {
                    dx[i] = xCurrent[i]-xPrev[i];
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
            WriteInVecCoord xinNew = Inherited::inDofs->writePositions();
     
            // New position of parents
            InVecCoord parentNew(xinNew.size());
            for(size_t i=0; i<xinNew.size();++i)
            {
                parentNew[i] = xinNew[i];
            }
   
            // Expected children positions: rotation from affine constraint
            WriteOutVecCoord xoutNew = Inherited::outDofs->writePositions();
            OutVecCoord expectedChildCoords(xoutNew.size());
  
            for(size_t i=0;i<xoutNew.size();++i)
            {
                OutFrame &f = expectedChildCoords[i].getF();
                f = testedRotation;
            }

           // run the mapping test
           return Inherited::runTest(parentInit,childInit,parentNew,expectedChildCoords);
               return true;

        }
        
    };

      // Define the list of DataTypes to instantiate
    using testing::Types;
    typedef Types<
       LinearMapping<Vec3Types, F321Types>,
       LinearMapping<Vec3Types, F331Types>,
       LinearMapping<Vec3Types, F332Types>,
       LinearMapping<Vec3Types, F311Types>
    > DataTypes; // the types to instantiate.

    // Test suite for all the instantiations
    TYPED_TEST_CASE(PointsDeformationMapping_test, DataTypes);

    // test case: polarcorotationalStrainMapping 
    TYPED_TEST( PointsDeformationMapping_test , VecDeformationMappingTest)
    {
        ASSERT_TRUE( this->runTest(1e-10));
    }

} // namespace sofa
