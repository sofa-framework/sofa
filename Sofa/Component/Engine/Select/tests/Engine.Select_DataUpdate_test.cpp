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

#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/component/engine/select/IndicesFromValues.h>
#include <sofa/component/engine/select/MergeROIs.h>
#include <sofa/component/engine/select/MeshROI.h>
#include <sofa/component/engine/select/MeshSampler.h>
#include <sofa/component/engine/select/MeshSubsetEngine.h>
#include <sofa/component/engine/select/PlaneROI.h>
#include <sofa/component/engine/select/PointsFromIndices.h>
#include <sofa/component/engine/select/ProximityROI.h>
#include <sofa/component/engine/select/SelectConnectedLabelsROI.h>
#include <sofa/component/engine/select/SelectLabelROI.h>
#include <sofa/component/engine/select/SphereROI.h>
#include <sofa/component/engine/select/SubsetTopology.h>
#include <sofa/component/engine/select/ValuesFromIndices.h>
#include <sofa/component/engine/select/ValuesFromPositions.h>

namespace sofa
{

typedef ::testing::Types<
//TestDataEngine< component::engine::select::BoxROI<defaulttype::Vec3Types> >, // getObject pb -> recuire a scene
//TestDataEngine< component::engine::select::PairBoxROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
//TestDataEngine< component::engine::select::PlaneROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
//TestDataEngine< component::engine::select::SphereROI<defaulttype::Vec3Types> >, // getObject pb -> require a scene
//TestDataEngine< component::engine::select::MeshSampler<defaulttype::Vec3Types> > // ???
TestDataEngine< component::engine::select::IndicesFromValues<int> >,
TestDataEngine< component::engine::select::MergeROIs >,
TestDataEngine< component::engine::select::MeshROI<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::select::MeshSubsetEngine<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::select::PointsFromIndices<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::select::ProximityROI<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::select::SelectConnectedLabelsROI<unsigned int> >,
TestDataEngine< component::engine::select::SelectLabelROI<unsigned int> >,
TestDataEngine< component::engine::select::SubsetTopology<defaulttype::Vec3Types> >,
TestDataEngine< component::engine::select::ValuesFromIndices<int> >,
TestDataEngine< component::engine::select::ValuesFromPositions<defaulttype::Vec3Types> >
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
