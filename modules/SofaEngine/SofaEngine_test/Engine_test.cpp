/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "TestEngine.h"
#include <SceneCreator/SceneCreator.h>
#include <SofaTest/Sofa_test.h>
#include <SofaTest/DataEngine_test.h>

#include <sofa/defaulttype/VecTypes.h>



namespace sofa {

/**  Test suite for engine using TestEngine class.
This class has a counter which shows how many times the update method is called. 
Add inputs to engine.
Check that the output is updated only if necessary.
For this test 3 engines are used.The output of engine1 is linked to the input of engine2 and also the input of engine3.
         engine2
        /
engine1
        \
         engine3

  */
struct Engine_test : public Sofa_test<>
{
    typedef sofa::component::engine::TestEngine TestEngine;
    TestEngine::SPtr engine1;
    TestEngine::SPtr engine2;
    TestEngine::SPtr engine3;

    /// Create the engines
    void SetUp()
    { 
       // Engine 1
       engine1 = sofa::core::objectmodel::New<TestEngine>();
       engine1->f_numberToMultiply.setValue(1);
       engine1->f_factor.setValue(2);
       engine1->init();
     
       // Engine 2 linked to the ouput of engine 1
       engine2 = sofa::core::objectmodel::New<TestEngine>();
       sofa::modeling::setDataLink(&engine1->f_result,&engine2->f_numberToMultiply);
       engine2->f_factor.setValue(3);
       engine2->init();

       // Engine 3 linked to the ouput of engine 1
       engine3 = sofa::core::objectmodel::New<TestEngine>();
       sofa::modeling::setDataLink(&engine1->f_result,&engine3->f_numberToMultiply);
       engine3->f_factor.setValue(3);
       engine3->init();

    }

    // Test if the output of engine2 is updated only if necessary
    void testUpdateEngine2()
    {
        //Get output engine2
       SReal result2 = engine2->f_result.getValue();
       result2 = engine2->f_result.getValue();

       // Test if update method of engine1 is called 1 time
       if(engine1->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine1 was called " << engine1->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine2 is called 1 time
       if(engine2->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine2 was called " << engine2->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine3 is not called
       if(engine3->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine3 must not be called " << std::endl;
       }

       // Test if result is correct
       ASSERT_EQ(result2,6);

    }

    // Test if the output of engine3 is updated only if necessary
    void testUpdateEngine3()
    {
        //Get output engine3
       SReal result3 = engine3->f_result.getValue();

       // Test if update method of engine1 is called 1 time
       if(engine1->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine1 was called " << engine1->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine2 is not called 
       if(engine2->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine2 must not be called " << std::endl;
       }

       // Test if update method of engine3 is called 1 time
       if(engine3->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine3 was called " << engine3->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if result is correct
       ASSERT_EQ(result3,6);

    }
    
    // Test the propagation: if the ouput is changed the input must not changed
    void testPropagationDirection()
    {
        // Check propagation direction

       // Change output value of engine3
       engine3->f_result.setValue(2);

       // Check that update methods are not called

       if(engine1->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine1 must not be called " << std::endl;
       }

       if(engine2->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine2 must not be called " << std::endl;
       }

       if(engine3->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine3 must not be called " << std::endl;
       }

       // Check that input value is not changed
       SReal input1 = engine1->f_numberToMultiply.getValue();

       ASSERT_EQ(input1,1);

    }

};

/// first test case: Check update method of engine2
TEST_F(Engine_test , check_engine2_update )
{
    this->testUpdateEngine2();
}

/// second test case: Check update method of engine3
TEST_F(Engine_test , check_engine3_update )
{
    this->testUpdateEngine3();
}

/// third test case: check propagation direction
TEST_F(Engine_test , check_propagation )
{
    this->testPropagationDirection();
}


}// namespace sofa




//////////////////////////

#include <SofaEngine/AverageCoord.h>
#include <SofaEngine/BoxROI.h>
#include <SofaEngine/PairBoxRoi.h>
#include <SofaEngine/PlaneROI.h>
#include <SofaEngine/SphereROI.h>
#include <SofaEngine/SelectLabelROI.h>
#include <SofaEngine/SelectConnectedLabelsROI.h>
#include <SofaEngine/DilateEngine.h>
#include <SofaEngine/GenerateCylinder.h>
#include <SofaEngine/ExtrudeSurface.h>
#include <SofaEngine/ExtrudeQuadsAndGenerateHexas.h>
#include <SofaEngine/ExtrudeEdgesAndGenerateQuads.h>
#include <SofaEngine/GenerateRigidMass.h>
#include <SofaEngine/GroupFilterYoungModulus.h>
#include <SofaEngine/MathOp.h>
#include <SofaEngine/MergeMeshes.h>
#include <SofaEngine/MergePoints.h>
#include <SofaEngine/MergeSets.h>
#include <SofaEngine/MergeVectors.h>
#include <SofaEngine/MergeROIs.h>
#include <SofaEngine/MeshBarycentricMapperEngine.h>
#include <SofaEngine/MeshROI.h>
#include <SofaEngine/TransformPosition.h>
#include <SofaEngine/TransformEngine.h>
#include <SofaEngine/TransformMatrixEngine.h>
#include <SofaEngine/PointsFromIndices.h>
#include <SofaEngine/ValuesFromIndices.h>
#include <SofaEngine/IndicesFromValues.h>
#include <SofaEngine/IndexValueMapper.h>
#include <SofaEngine/ROIValueMapper.h>
#include <SofaEngine/JoinPoints.h>
#include <SofaEngine/MapIndices.h>
#include <SofaEngine/RandomPointDistributionInSurface.h>
#include <SofaEngine/SmoothMeshEngine.h>
#include <SofaEngine/Spiral.h>
#include <SofaEngine/Vertex2Frame.h>
#include <SofaEngine/TextureInterpolation.h>
#include <SofaEngine/SubsetTopology.h>
#include <SofaEngine/RigidToQuatEngine.h>
#include <SofaEngine/QuatToRigidEngine.h>
#include <SofaEngine/ValuesFromPositions.h>
#include <SofaEngine/NormalsFromPoints.h>
#include <SofaEngine/ClusteringEngine.h>
#include <SofaEngine/ShapeMatching.h>
#include <SofaEngine/ProximityROI.h>
#include <SofaEngine/HausdorffDistance.h>
#include <SofaEngine/NormEngine.h>
#include <SofaEngine/MeshClosingEngine.h>
#include <SofaEngine/MeshSubsetEngine.h>
#include <SofaEngine/MeshSampler.h>

namespace sofa {


// testing every engines of SofaEngine here

typedef testing::Types<
//TestDataEngine< component::engine::AverageCoord<defaulttype::Vec3Types> >,  // getObject pb -> require a scene
//TestDataEngine< component::engine::BoxROI<defaulttype::Vec3Types> >, // getObject pb -> recuire a scene
//TestDataEngine< component::engine::PairBoxROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
//TestDataEngine< component::engine::PlaneROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
//TestDataEngine< component::engine::SphereROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
TestDataEngine< component::engine::SelectLabelROI<unsigned int> >,
TestDataEngine< component::engine::SelectConnectedLabelsROI<unsigned int> >,
TestDataEngine< component::engine::DilateEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::GenerateCylinder<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ExtrudeSurface<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ExtrudeQuadsAndGenerateHexas<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ExtrudeEdgesAndGenerateQuads<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::GenerateRigidMass<defaulttype::Rigid3Types,defaulttype::Rigid3Mass> >,
TestDataEngine< component::engine::GroupFilterYoungModulus<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::MathOp< helper::vector<int> > >,
TestDataEngine< component::engine::MergeMeshes<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::MergePoints<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::MergeSets<int> >,
TestDataEngine< component::engine::MergeVectors< helper::vector<defaulttype::Vector3> > >,
TestDataEngine< component::engine::MergeROIs >,
//TestDataEngine< component::engine::MeshBarycentricMapperEngine<defaulttype::Vec3Types> >, // require a scene
TestDataEngine< component::engine::MeshROI<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::TransformPosition<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::TransformEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::TranslateTransformMatrixEngine >,
TestDataEngine< component::engine::PointsFromIndices<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ValuesFromIndices<int> >,
TestDataEngine< component::engine::IndicesFromValues<int> >,
TestDataEngine< component::engine::IndexValueMapper<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ROIValueMapper >,
TestDataEngine< component::engine::JoinPoints<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::MapIndices<int> >,
TestDataEngine< component::engine::RandomPointDistributionInSurface<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::SmoothMeshEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::Spiral<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::Vertex2Frame<defaulttype::Rigid3Types> >,
TestDataEngine< component::engine::TextureInterpolation<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::SubsetTopology<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::RigidToQuatEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::QuatToRigidEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ValuesFromPositions<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::NormalsFromPoints<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::ClusteringEngine<defaulttype::Vec3Types> >,
//TestDataEngine< component::engine::ShapeMatching<defaulttype::Vec3Types> >, // getObject pb -> require a scene
TestDataEngine< component::engine::ProximityROI<defaulttype::Vec3Types> >,
//TestDataEngine< component::engine::HausdorffDistance<defaulttype::Vec3Types> >, // ???
TestDataEngine< component::engine::NormEngine<defaulttype::Vector3> >,
TestDataEngine< component::engine::MeshClosingEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::MeshSubsetEngine<defaulttype::Vec3Types> >
//TestDataEngine< component::engine::MeshSampler<defaulttype::Vec3Types> > // ???
> TestTypes; // the types to instanciate.


//// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(DataEngine_test, TestTypes);

//// test number of call to DataEngine::update
TYPED_TEST( DataEngine_test , basic_test )
{
    this->run_basic_test();
}

}// namespace sofa
