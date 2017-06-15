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
#include <sofa/helper/Quater.h>

// Including component
#include "../deformationMapping/LinearMapping.h"

#include <SofaTest/Mapping_test.h>
//#include "../shapeFunction/DiffusionShapeFunction.h"
namespace sofa {

    using namespace defaulttype;
    using namespace component;
    using namespace component::mapping;


   /// Linear Deformation mappings test
   /**  
    * Test the deformation mappings by applying any affine transformation  
    * to all dofs and checks if the resulting deformation gradient is equal to the affine transformation.
   */

    template <typename _Mapping>
    struct AffineLinearDeformationMappings_test : public Mapping_test<_Mapping>
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

        /// Tested quaternion: random quaternion
        Quat testedQuaternion;
        /// Tested Rotation: matrix from testedQuaternion 
        defaulttype::Mat<3,3,Real> testedRotation; 
        /// Tested Translation: random translation
        Vec3 testedTranslation;

        // Constructor: call the constructor of the base class which loads the scene to test
        AffineLinearDeformationMappings_test() : Mapping_test<_Mapping>(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "AffineLineDeformationMapping.scn")
        {   
            Inherited::errorMax = 5000;
            // Set random rotation and translation
            this->SetRandomAffineTransform();
        }
             
        void SetRandomAffineTransform ()
        {
            // Matrix 3*3
            for( int j=0; j<testedRotation.nbCols; j++)
            {
                for( int i=0; i<testedRotation.nbLines; i++)
                {
                    // random value between -1 and 1
                    testedRotation(i,j)=helper::drand(1);
                }
            }

            // Translation
            for(size_t i=0;i<testedTranslation.size();++i)
            {
                // Random value between -2 and 2
                testedTranslation[i]=Real(helper::drand(2));
            }

        }

        // Apply affine transform to each dof
        void applyAffineTransform(InVecCoord& x0, InVecCoord& xf)
        {
            for (size_t i=0; i < x0.size() ; ++i)
            {
                // Translation
                xf[i].getCenter() = testedRotation*(x0[i].getCenter()) + testedTranslation;
     
                // Rotation
                xf[i].getAffine() = testedRotation*x0[i].getAffine();

            }
        }
        
        using Inherited::runTest;
        /// After simulation compare the positions of points to the theoretical positions.
        bool runTest(double /*convergenceAccuracy*/)
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(this->root.get());
     
            // xin
            typename  InDOFs::ReadVecCoord x = this->inDofs->readPositions();
            InVecCoord xin(x.size());
            copyFromData(xin,x);
    
            // xout
            typename  OutDOFs::ReadVecCoord xelasticityDofs = this->outDofs->readPositions();
            OutVecCoord xout(xelasticityDofs.size());
            copyFromData(xout,xelasticityDofs);

            // Apply affine transform to each dof
            InVecCoord parentNew(xin.size());
            this->applyAffineTransform(xin,parentNew);

            // Expected children positions: rotation from affine constraint
            OutVecCoord expectedChildCoords(xout.size());
  
            for(size_t i=0;i<xout.size();++i)
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
    typedef testing::Types<
        LinearMapping<Affine3Types, F331Types>,
        LinearMapping<Affine3Types, F332Types>
    > DataTypes; // the types to instantiate.

    // Test suite for all the instantiations
    TYPED_TEST_CASE(AffineLinearDeformationMappings_test, DataTypes);

    // test case: polarcorotationalStrainMapping 
    TYPED_TEST( AffineLinearDeformationMappings_test , AffineStrainDeformationPatchTest)
    {
        ASSERT_TRUE( this->runTest(1e-15));
    }

} // namespace sofa
