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
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/linear/SubsetMultiMapping.h>

#include <sofa/component/mapping/testing/MultiMappingTestCreation.h>

namespace sofa {

using component::mapping::linear::SubsetMultiMapping;

/**
 * Test suite for SubsetMultiMapping.
 */
template <typename _SubsetMultiMapping>
struct SubsetMultiMappingTest : public MultiMapping_test<_SubsetMultiMapping>
{
    using In = typename _SubsetMultiMapping::In;
    using Out = typename _SubsetMultiMapping::Out;

    /**
     * Two parent particles, two children
     */
    bool test_two_parents_one_child()
    {
        static constexpr int NP = 2;
        this->setupScene(NP); // NP parents, 1 child
        _SubsetMultiMapping* smm = static_cast<_SubsetMultiMapping*>( this->mapping );

        // parent positions
        type::vector< VecCoord_t<In> > incoords(NP);
        for (int i = 0; i < NP; i++)
        {
            incoords[i].resize(1);
            In::set( incoords[i][0], i+1.,-2., 3. );
        }

        // subset
        smm->addPoint(smm->getMechFrom()[0],0);  // parent, index in parent
        smm->addPoint(smm->getMechFrom()[1],0);  // parent, index in parent

        // expected child positions
        VecCoord_t<Out> outcoords(2);
        Out::set( outcoords[0], 1.  , -2., 3. );
        Out::set( outcoords[1], 1+1., -2., 3. );

        return this->runTest(incoords,outcoords);
    }
};

using ::testing::Types;
typedef Types<
    SubsetMultiMapping<defaulttype::Vec3Types,defaulttype::Vec3Types>,
    SubsetMultiMapping<defaulttype::Vec2Types,defaulttype::Vec2Types>,
    SubsetMultiMapping<defaulttype::Vec1Types,defaulttype::Vec1Types>,
    SubsetMultiMapping<defaulttype::Rigid3Types,defaulttype::Rigid3Types>,
    SubsetMultiMapping<defaulttype::Rigid3Types,defaulttype::Vec3Types>
> DataTypes;

TYPED_TEST_SUITE(SubsetMultiMappingTest, DataTypes);

TYPED_TEST( SubsetMultiMappingTest , two_parents_one_child )
{
    ASSERT_TRUE(this->test_two_parents_one_child());
}

} // namespace sofa
