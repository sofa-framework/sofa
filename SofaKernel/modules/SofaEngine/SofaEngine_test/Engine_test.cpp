/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "TestEngine.h"
#include <SceneCreator/SceneCreator.h>
#include <SofaTest/Sofa_test.h>
#include <SofaTest/DataEngine_test.h>

#include <sofa/defaulttype/VecTypes.h>


#include <SofaTest/TestMessageHandler.h>


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
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testUpdateEngine2();
}

/// second test case: Check update method of engine3
TEST_F(Engine_test , check_engine3_update )
{
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testUpdateEngine3();
}

/// third test case: check propagation direction
TEST_F(Engine_test , check_propagation )
{
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testPropagationDirection();
}


}// namespace sofa




//////////////////////////

#include <SofaGeneralEngine/AverageCoord.h>
#include <SofaEngine/BoxROI.h>
#include <SofaGeneralEngine/PairBoxRoi.h>
#include <SofaGeneralEngine/PlaneROI.h>
#include <SofaGeneralEngine/SphereROI.h>
#include <SofaGeneralEngine/SelectLabelROI.h>
#include <SofaGeneralEngine/SelectConnectedLabelsROI.h>
#include <SofaGeneralEngine/DilateEngine.h>
#include <SofaGeneralEngine/GenerateCylinder.h>
#include <SofaGeneralEngine/ExtrudeSurface.h>
#include <SofaGeneralEngine/ExtrudeQuadsAndGenerateHexas.h>
#include <SofaGeneralEngine/ExtrudeEdgesAndGenerateQuads.h>
#include <SofaGeneralEngine/GenerateRigidMass.h>
#include <SofaGeneralEngine/GroupFilterYoungModulus.h>
#include <SofaGeneralEngine/MathOp.h>
#include <SofaGeneralEngine/MergeMeshes.h>
#include <SofaGeneralEngine/MergePoints.h>
#include <SofaGeneralEngine/MergeSets.h>
#include <SofaGeneralEngine/MergeVectors.h>
#include <SofaGeneralEngine/MergeROIs.h>
#include <SofaGeneralEngine/MeshBarycentricMapperEngine.h>
#include <SofaGeneralEngine/MeshROI.h>
#include <SofaGeneralEngine/TransformPosition.h>
#include <SofaGeneralEngine/TransformEngine.h>
#include <SofaGeneralEngine/TransformMatrixEngine.h>
#include <SofaGeneralEngine/PointsFromIndices.h>
#include <SofaGeneralEngine/ValuesFromIndices.h>
#include <SofaGeneralEngine/IndicesFromValues.h>
#include <SofaGeneralEngine/IndexValueMapper.h>
#include <SofaGeneralEngine/ROIValueMapper.h>
#include <SofaGeneralEngine/JoinPoints.h>
#include <SofaGeneralEngine/MapIndices.h>
#include <SofaGeneralEngine/RandomPointDistributionInSurface.h>
#include <SofaGeneralEngine/SmoothMeshEngine.h>
#include <SofaGeneralEngine/Spiral.h>
#include <SofaGeneralEngine/Vertex2Frame.h>
#include <SofaGeneralEngine/TextureInterpolation.h>
#include <SofaGeneralEngine/SubsetTopology.h>
#include <SofaGeneralEngine/RigidToQuatEngine.h>
#include <SofaGeneralEngine/QuatToRigidEngine.h>
#include <SofaGeneralEngine/ValuesFromPositions.h>
#include <SofaGeneralEngine/NormalsFromPoints.h>
#include <SofaGeneralEngine/ClusteringEngine.h>
#include <SofaGeneralEngine/ShapeMatching.h>
#include <SofaGeneralEngine/ProximityROI.h>
#include <SofaGeneralEngine/HausdorffDistance.h>
#include <SofaGeneralEngine/NormEngine.h>
#include <SofaGeneralEngine/MeshClosingEngine.h>
#include <SofaGeneralEngine/MeshSubsetEngine.h>
#include <SofaGeneralEngine/MeshSampler.h>
#include <SofaGeneralEngine/SumEngine.h>
#include <SofaGeneralEngine/DifferenceEngine.h>

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
#ifdef SOFA_WITH_DOUBLE
TestDataEngine< component::engine::DilateEngine<defaulttype::Vec3Types> >, // DilateEngine only defined for Vec3dTypes
#endif
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
TestDataEngine< component::engine::MeshSubsetEngine<defaulttype::Vec3Types> >,
//TestDataEngine< component::engine::MeshSampler<defaulttype::Vec3Types> > // ???
TestDataEngine< component::engine::SumEngine<defaulttype::Vector3> >,
TestDataEngine< component::engine::DifferenceEngine<defaulttype::Vector3> >
> TestTypes; // the types to instanciate.


//// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(DataEngine_test, TestTypes);

//// test number of call to DataEngine::update
TYPED_TEST( DataEngine_test , basic_test )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->run_basic_test();
}

}// namespace sofa
