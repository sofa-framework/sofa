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
#include <sofa/component/mapping/nonlinear/DistanceMultiMapping.h>
#include <sofa/component/mapping/testing/MultiMappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>

namespace sofa
{
template <typename _DistanceMultiMapping>
struct DistanceMultiMappingTest : public MultiMapping_test<_DistanceMultiMapping>
{
    using In = typename _DistanceMultiMapping::In;
    using Out = typename _DistanceMultiMapping::Out;

    bool test(bool computeDistance)
    {
        constexpr int numberOfParents = 2;
        this->setupScene(numberOfParents); // NP parents, 1 child

        _DistanceMultiMapping* map = static_cast<_DistanceMultiMapping*>( this->mapping );
        EXPECT_NE(map, nullptr);

        map->d_indexPairs.setValue({{0, 0}, {1, 0}});
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);
        map->d_computeDistance.setValue(computeDistance);

        const component::topology::container::dynamic::EdgeSetTopologyContainer::SPtr edges = sofa::core::objectmodel::New<component::topology::container::dynamic::EdgeSetTopologyContainer>();
        this->root->addObject(edges);
        edges->addEdge( 0, 1 );

        // parent positions
        sofa::type::vector< VecCoord_t<In> > incoords(numberOfParents);
        for (auto& in : incoords)
        {
            in.resize(1);
        }

        In::set( incoords[0][0], 1, 2, 3);
        In::set( incoords[1][0], 4, 1, 5);

        VecCoord_t<Out> outcoords(1);
        if (computeDistance)
        {
            const auto expectedDistance = (In::getCPos(incoords[1][0]) - In::getCPos(incoords[0][0])).norm();
            Out::set( outcoords[0], expectedDistance, 0.,0.);
        }
        else
        {
            Out::set( outcoords[0], 0,0,0);
        }
        return this->runTest(incoords,outcoords);
    }
};

using ::testing::Types;
typedef Types<
    sofa::component::mapping::nonlinear::DistanceMultiMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>,
    sofa::component::mapping::nonlinear::DistanceMultiMapping<defaulttype::Rigid3Types,defaulttype::Vec1Types>
> DataTypes;


TYPED_TEST_SUITE(DistanceMultiMappingTest, DataTypes);

TYPED_TEST( DistanceMultiMappingTest, computeDistance )
{
    this->errorMax = 1000;
    ASSERT_TRUE(this->test(true));
}

TYPED_TEST( DistanceMultiMappingTest, notComputeDistance )
{
    this->errorMax = 1000;
    ASSERT_TRUE(this->test(false));
}

}

