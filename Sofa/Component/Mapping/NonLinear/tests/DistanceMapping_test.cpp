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

#include <sofa/component/mapping/nonlinear/DistanceMapping.h>

#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>

namespace sofa {
namespace {



/**  Test suite for DistanceMapping.
 *
 * @author Matthieu Nesme
  */
template <typename DistanceMapping>
struct DistanceMappingTest : public sofa::mapping_test::Mapping_test<DistanceMapping>
{
    typedef typename DistanceMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename DistanceMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test()
    {
        DistanceMapping* map = static_cast<DistanceMapping*>( this->mapping );
        map->f_computeDistance.setValue(true);
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);

        const component::topology::container::dynamic::EdgeSetTopologyContainer::SPtr edges = sofa::core::objectmodel::New<component::topology::container::dynamic::EdgeSetTopologyContainer>();
        this->root->addObject(edges);
        edges->addEdge( 0, 1 );

        // parent positions
        InVecCoord incoord(2);
        InDataTypes::set( incoord[0], 0,0,0 );
        InDataTypes::set( incoord[1], 1,1,1 );

        // expected child positions
        OutVecCoord expectedoutcoord;
        expectedoutcoord.push_back( type::Vec1( std::sqrt(3.0) ) );

        return this->runTest( incoord, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using ::testing::Types;
typedef Types<
component::mapping::nonlinear::DistanceMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>,
component::mapping::nonlinear::DistanceMapping<defaulttype::Rigid3Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE( DistanceMappingTest, DataTypes );

// test case
TYPED_TEST( DistanceMappingTest , test )
{
    ASSERT_TRUE(this->test());
}

} // namespace
} // namespace sofa
