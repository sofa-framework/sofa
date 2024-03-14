/******************************************************************************
 *                 SOFA, Simulation Open-Framework Architecture                *
 *                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/component/engine/testing/DataEngineTestCreation.h>
#include <sofa/defaulttype/VecTypes.h>

//////////////////////////

#include <sofa/component/engine/generate/ExtrudeEdgesAndGenerateQuads.h>
#include <sofa/component/engine/generate/ExtrudeQuadsAndGenerateHexas.h>
#include <sofa/component/engine/generate/ExtrudeSurface.h>
#include <sofa/component/engine/generate/GenerateCylinder.h>
#include <sofa/component/engine/generate/GenerateGrid.h>
#include <sofa/component/engine/generate/GenerateRigidMass.h>
#include <sofa/component/engine/generate/GenerateSphere.h>
#include <sofa/component/engine/generate/GroupFilterYoungModulus.h>
#include <sofa/component/engine/generate/JoinPoints.h>
#include <sofa/component/engine/generate/MergeMeshes.h>
#include <sofa/component/engine/generate/MergePoints.h>
#include <sofa/component/engine/generate/MergeSets.h>
#include <sofa/component/engine/generate/MergeVectors.h>
#include <sofa/component/engine/generate/MeshBarycentricMapperEngine.h>
#include <sofa/component/engine/generate/MeshClosingEngine.h>
#include <sofa/component/engine/generate/NormEngine.h>
#include <sofa/component/engine/generate/NormalsFromPoints.h>
#include <sofa/component/engine/generate/RandomPointDistributionInSurface.h>
#include <sofa/component/engine/generate/Spiral.h>

namespace sofa
{
// specialization for special cases
template <>
void DataEngine_test<
    TestDataEngine<component::engine::generate::JoinPoints<defaulttype::Vec3Types> > >::preInit()
{
    m_engineInput->findData("points")->read("0.0 0.0 0.0");
}

template <>
void DataEngine_test<TestDataEngine<
    component::engine::generate::RandomPointDistributionInSurface<defaulttype::Vec3Types> > >::preInit()
{
    m_engineInput->findData("vertices")->read("-0.5 -0.5 -0.5  1 0 0  0 1 0  0 0 1");
    m_engineInput->findData("triangles")->read("0 2 1  0 1 3  0 3 2   1 2 3");
}

typedef ::testing::Types<
//TestDataEngine< component::engine::generate::MeshBarycentricMapperEngine<defaulttype::Vec3Types> >, // require a scene
TestDataEngine< component::engine::generate::ExtrudeEdgesAndGenerateQuads<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::ExtrudeQuadsAndGenerateHexas<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::ExtrudeSurface<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::GenerateCylinder<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::GenerateRigidMass<defaulttype::Rigid3Types,defaulttype::Rigid3Mass> >,
TestDataEngine< component::engine::generate::GroupFilterYoungModulus<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::JoinPoints<defaulttype::Vec3Types> >, 
TestDataEngine< component::engine::generate::MergeMeshes<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::MergePoints<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::MergeSets<int> >,
TestDataEngine< component::engine::generate::MergeVectors< type::vector<type::Vec3> > >,
TestDataEngine< component::engine::generate::MeshClosingEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::NormEngine<type::Vec3> >,
TestDataEngine< component::engine::generate::NormalsFromPoints<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::RandomPointDistributionInSurface<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::generate::Spiral<defaulttype::Vec3Types> >
> TestTypes; // the types to instanciate.

//// ========= Tests to run for each instanciated type
TYPED_TEST_SUITE(DataEngine_test, TestTypes);

//// test number of call to DataEngine::update
TYPED_TEST(DataEngine_test, basic_test)
{
    EXPECT_MSG_NOEMIT(Error);
    this->run_basic_test();
}

}  // namespace sofa
