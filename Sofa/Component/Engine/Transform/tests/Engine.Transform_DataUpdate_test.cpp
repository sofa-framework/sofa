
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

#include <sofa/component/engine/transform/DifferenceEngine.h>
#include <sofa/component/engine/transform/DilateEngine.h>
#include <sofa/component/engine/transform/IndexValueMapper.h>
#include <sofa/component/engine/transform/MapIndices.h>
#include <sofa/component/engine/transform/MathOp.h>
#include <sofa/component/engine/transform/QuatToRigidEngine.h>
#include <sofa/component/engine/transform/ROIValueMapper.h>
#include <sofa/component/engine/transform/RigidToQuatEngine.h>
#include <sofa/component/engine/transform/SmoothMeshEngine.h>
#include <sofa/component/engine/transform/TransformEngine.h>
#include <sofa/component/engine/transform/TransformMatrixEngine.h>
#include <sofa/component/engine/transform/TransformPosition.h>
#include <sofa/component/engine/transform/Vertex2Frame.h>

namespace sofa
{

typedef ::testing::Types<
TestDataEngine< component::engine::transform::DifferenceEngine<type::Vec3> >,
TestDataEngine< component::engine::transform::DilateEngine<defaulttype::Vec3Types> >, // DilateEngine only defined for Vec3dTypes
TestDataEngine< component::engine::transform::IndexValueMapper<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::MapIndices<int> >,
TestDataEngine< component::engine::transform::MathOp< type::vector<int> > >,
TestDataEngine< component::engine::transform::QuatToRigidEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::ROIValueMapper >,
TestDataEngine< component::engine::transform::RigidToQuatEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::SmoothMeshEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::TransformEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::TransformPosition<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::transform::TranslateTransformMatrixEngine >,
TestDataEngine< component::engine::transform::Vertex2Frame<defaulttype::Rigid3Types> >
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
