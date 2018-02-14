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
#include <SofaTest/Mapping_test.h>
#include <sofa/helper/Quater.h>
#include <image/ImageTypes.h>
#include <image/ImageContainer.h>
#include "../shapeFunction/VoronoiShapeFunction.h"
#include "../shapeFunction/ShepardShapeFunction.h"
#include "../shapeFunction/HatShapeFunction.h"
#include "../shapeFunction/ShapeFunctionDiscretizer.h"
#include "../shapeFunction/DiffusionShapeFunction.h"
#include "../types/AffineTypes.h"
#include "../types/DeformationGradientTypes.h"

// Including component
#include "../deformationMapping/LinearMapping.h"

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
    struct ShapeFunction_test : public Mapping_test<_Mapping>
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
        ShapeFunction_test() : Mapping_test<_Mapping>(std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + "ShapeFunctionTest.scn")
        {   
            if( _Mapping::Out::Basis::order >= 1 ) // F332Types
                this->deltaRange = std::make_pair( 1e5, 1e9 );

            this->errorMax = this->deltaRange.second * 5;

            // Set random rotation and translation
            this->SetRandomAffineTransform();
        }
             
        void SetShapeFunction (int shapeFunctionCase)
        {
            simulation::Node::SPtr patchNode = this->root->getChild("Patch");

            // Complete the scene
            typedef core::behavior::ShapeFunctionTypes<3, SReal> ShapeFunctionType;

            // Get Image container
            typedef typename component::container::ImageContainer<ImageUC> ImageContainer;
            core::objectmodel::BaseContext* rootNode = this->root.get();
            typename ImageContainer::SPtr imageContainerSptr = rootNode->get<ImageContainer >(sofa::core::objectmodel::BaseContext::SearchDown);

            // Voronoi Shape Function
            if(shapeFunctionCase == 0)
            {
                typedef component::shapefunction::VoronoiShapeFunction<ShapeFunctionType,ImageUC> VoronoiShapeFunction;
                VoronoiShapeFunction::SPtr voronoiShapeFctSptr = modeling::addNew <VoronoiShapeFunction> (patchNode,"shapeFunction");
                sofa::modeling::setDataLink(&Inherited::inDofs->x0,&voronoiShapeFctSptr->f_position);

                //std::cout << "voronoiShapeFctSptr->f_position size = " << voronoiShapeFctSptr->f_position.getValue().size() << std::endl;


                voronoiShapeFctSptr->setSrc("@"+imageContainerSptr->getName(), imageContainerSptr.get());

                voronoiShapeFctSptr->useDijkstra.setValue(1);
                voronoiShapeFctSptr->method.setValue(0);
                voronoiShapeFctSptr->f_nbRef.setValue(10);
            }
         
            // Diffusion shape function
            else if (shapeFunctionCase == 1)
            {
                typedef component::shapefunction::DiffusionShapeFunction<ShapeFunctionType,ImageUC> DiffusionShapeFunction;
                DiffusionShapeFunction::SPtr diffusionShapeFctSptr = modeling::addNew <DiffusionShapeFunction> (patchNode,"shapeFunction");
                sofa::modeling::setDataLink(&Inherited::inDofs->x0,&diffusionShapeFctSptr->f_position);
                diffusionShapeFctSptr->setSrc("@"+imageContainerSptr->getName(), imageContainerSptr.get());

            }
           
            // Shepard shape function
            else if (shapeFunctionCase == 2)
            {
                typedef component::shapefunction::ShepardShapeFunction<ShapeFunctionType> ShepardShapeFunction;
                ShepardShapeFunction::SPtr shepardShapeFctSptr = modeling::addNew <ShepardShapeFunction> (patchNode,"shapeFunction");
                sofa::modeling::setDataLink(&Inherited::inDofs->x0,&shepardShapeFctSptr->f_position);
                shepardShapeFctSptr->power.setValue(2);
                shepardShapeFctSptr->f_nbRef.setValue(10);
            }
            
            // Hat shape function with shape function discretizer
            else if (shapeFunctionCase == 3)
            {
                typedef component::shapefunction::HatShapeFunction<ShapeFunctionType> HatShapeFunction;
                HatShapeFunction::SPtr hatShapeFctSptr = modeling::addNew <HatShapeFunction> (patchNode,"shapeFunction");
                sofa::modeling::setDataLink(&Inherited::inDofs->x0,&hatShapeFctSptr->f_position);
                hatShapeFctSptr->f_nbRef.setValue(10);
                typedef component::engine::ShapeFunctionDiscretizer<ImageUC> ShapeFunctionDiscretizer;
                ShapeFunctionDiscretizer::SPtr shapeFunctionDiscretizerSPtr = modeling::addNew <ShapeFunctionDiscretizer> (patchNode,"shapeFunctionDiscretizer");
                shapeFunctionDiscretizerSPtr->setSrc("@"+imageContainerSptr->getName(), imageContainerSptr.get());
            }

        }
        
        void SetRandomAffineTransform ()
        {
            // Matrix 3*3
            for( int j=0; j<testedRotation.nbCols; j++)
            {
                for( int i=0; i<testedRotation.nbLines; i++)
                {
                    testedRotation(i,j)= helper::drand(1);
                }
            }

            // Translation
            for(size_t i=0;i<testedTranslation.size();++i)
            {
                testedTranslation[i]=helper::drand(2);
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
        bool runTest()
        {
            // Init simulation
            sofa::simulation::getSimulation()->init(this->root.get());
     
            // xin
            typename  InDOFs::ReadVecCoord x = Inherited::inDofs->readPositions();
            InVecCoord xin(x.size());
            copyFromData(xin,x);

            // xout
            typename  OutDOFs::ReadVecCoord xelasticityDofs = Inherited::outDofs->readPositions();
            OutVecCoord xout(xelasticityDofs.size());
            copyFromData(xout,xelasticityDofs);

            typename  OutDOFs::ReadVecCoord xelasticityDofs_rest = Inherited::outDofs->readRestPositions();
            OutVecCoord xout_rest(xelasticityDofs_rest.size());
            copyFromData(xout_rest,xelasticityDofs_rest);

            // Apply affine transform to each dof
            InVecCoord parentNew(xin.size());
            this->applyAffineTransform(xin,parentNew);

            // Expected children positions: rotation from affine constraint
            OutVecCoord expectedChildCoords(xout.size());
  
            for(size_t i=0;i<xout.size();++i)
            {
                OutFrame &f = expectedChildCoords[i].getF();
                f = testedRotation*xelasticityDofs_rest[i].getF();
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
    TYPED_TEST_CASE(ShapeFunction_test, DataTypes);

    // test case: voronoi shape function test
    TYPED_TEST( ShapeFunction_test , VoronoiShapeFunctionTest)
    {
        this->SetShapeFunction(0);
        ASSERT_TRUE( this->runTest());
    }

    // test case: diffusion shape function test
    TYPED_TEST( ShapeFunction_test , DiffusionShapeFunctionTest)
    {
        this->SetShapeFunction(1);
        ASSERT_TRUE( this->runTest());
    }
    
    // test case: shepard shape function test
    TYPED_TEST( ShapeFunction_test , ShepardShapeFunctionTest)
    {
        this->SetShapeFunction(2);
        ASSERT_TRUE( this->runTest());
    }

    // test case: hat shape function test
    TYPED_TEST( ShapeFunction_test , HatShapeFunctionTest)
    {
        this->SetShapeFunction(3);
        ASSERT_TRUE( this->runTest());
    }

} // namespace sofa
