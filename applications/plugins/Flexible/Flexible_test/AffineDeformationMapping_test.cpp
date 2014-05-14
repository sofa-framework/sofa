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
#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

// Including component
#include "../deformationMapping/LinearMapping.h"

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
        Coord testedTranslation;
        /// Seed for random value
        long seed;
        /// Random generator
        sofa::helper::RandomGenerator randomGenerator;

        // Constructor: call the constructor of the base class which loads the scene to test
        AffineLinearDeformationMappings_test() : Mapping_test<_Mapping>(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "AffineLineDeformationMapping.scn")
        {   
            Inherited::errorMax = 5000;
            seed=2;
            // Set random rotation and translation
            randomGenerator.initSeed(seed);
            this->SetRandomTestedRotationAndTranslation(seed);
        }
             
        void SetRandomTestedRotationAndTranslation(int)
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
            testedQuaternion.toMatrix(testedRotation);

            // Translation
            for(size_t i=0;i<Coord::total_size;++i)
            {
                if(i<3)
                    testedTranslation[i]=randomGenerator.random<SReal>(-2.0,2.0);
                else
                    testedTranslation[i]=testedQuaternion[i]; 
            }

        }

        // Apply affine transform to each dof
        void applyAffineTransform(InVecCoord& x0, InVecCoord& xf)
        {
            for (size_t i=0; i < x0.size() ; ++i)
            {
                // Translation
                xf[i].getCenter() = testedRotation*(x0[i].getCenter()) + testedTranslation.getCenter();
     
                // Rotation
                xf[i].getAffine() = testedRotation*x0[i].getAffine();

            }
        }
        
        /// After simulation compare the positions of points to the theoretical positions.
        bool runTest(double convergenceAccuracy)
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
    typedef Types<
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