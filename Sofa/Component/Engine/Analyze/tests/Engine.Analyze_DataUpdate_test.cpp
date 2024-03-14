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

#include <sofa/component/engine/analyze/AverageCoord.h>
#include <sofa/component/engine/analyze/ClusteringEngine.h>
#include <sofa/component/engine/analyze/HausdorffDistance.h>
#include <sofa/component/engine/analyze/ShapeMatching.h>
#include <sofa/component/engine/analyze/SumEngine.h>


namespace sofa
{

typedef ::testing::Types<
//TestDataEngine< component::engine::analyze::AverageCoord<defaulttype::Vec3Types> >,  // getObject pb -> require a scene
//TestDataEngine< component::engine::analyze::HausdorffDistance<defaulttype::Vec3Types> >, // ???
//TestDataEngine< component::engine::analyze::ShapeMatching<defaulttype::Vec3Types> >, // getObject pb -> require a scene
TestDataEngine< component::engine::analyze::ClusteringEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::analyze::SumEngine<type::Vec3> >
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
