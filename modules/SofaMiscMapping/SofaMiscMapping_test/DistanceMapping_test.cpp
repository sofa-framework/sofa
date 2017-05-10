/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaTest/Mapping_test.h>
#include <SofaTest/MultiMapping_test.h>
#include <SofaMiscMapping/DistanceMapping.h>


namespace sofa {
namespace {



/**  Test suite for DistanceMapping.
 *
 * @author Matthieu Nesme
  */
template <typename DistanceMapping>
struct DistanceMappingTest : public Mapping_test<DistanceMapping>
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
        map->d_geometricStiffness.setValue(1);

        {
        typename DistanceMapping::VecPair pairs(2);
        pairs[0].set(0,1);
        pairs[1].set(2,1);
        map->d_pairs.setValue(pairs);
        }



        // parent positions
        InVecCoord incoord(3);
        InDataTypes::set( incoord[0], 0,0,0 );
        InDataTypes::set( incoord[1], 1,1,1 );
        InDataTypes::set( incoord[2], 6,3,-1 );

        // expected child positions
        OutVecCoord expectedoutcoord;
        expectedoutcoord.push_back( defaulttype::Vector1( std::sqrt(3.0) ) );
        expectedoutcoord.push_back( defaulttype::Vector1( std::sqrt(33.0) ) );

        return this->runTest( incoord, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
component::mapping::DistanceMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>,
component::mapping::DistanceMapping<defaulttype::Rigid3Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE( DistanceMappingTest, DataTypes );

// test case
TYPED_TEST( DistanceMappingTest , test )
{
    ASSERT_TRUE(this->test());
}



//////////////



/**  Test suite for DistanceMultiMapping.
 *
 * @author Matthieu Nesme
 * @date 2017
 */
template <typename DistanceMultiMapping>
struct DistanceMultiMappingTest : public MultiMapping_test<DistanceMultiMapping>
{
    typedef typename DistanceMultiMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename DistanceMultiMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test(unsigned nbParents)
    {
        this->setupScene(nbParents); // nbParents parents, 1 child

        DistanceMultiMapping* map = static_cast<DistanceMultiMapping*>( this->mapping );
        map->f_computeDistance.setValue(true);
        map->d_geometricStiffness.setValue(1);


        // parent positions
        helper::vector< InVecCoord > incoords(nbParents);

        // expected child positions
        OutVecCoord expectedoutcoord(nbParents*(nbParents-1)*.5); // link them all together

        typename DistanceMultiMapping::VecPair pairs;
        unsigned nb=0;
        for( unsigned i=0; i<nbParents; i++ )
        {
            incoords[i].resize(1);
            InDataTypes::set( incoords[i][0], i,i,i );

            for( unsigned j=0;j<i;++j)
            {
                typename DistanceMultiMapping::Pair p;
                p[0][0] = j; p[0][1] = 0;
                p[1][0] = i; p[1][1] = 0;
                pairs.push_back(p);

                expectedoutcoord[nb++][0] = std::sqrt(3.0*(i-j)*(i-j));
            }
        }

//        msg_info("DistanceMultiMappingTest")<<"edges:"<<edges->d_edge;

        map->d_pairs.setValue(pairs);

        return this->runTest( incoords, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
component::mapping::DistanceMultiMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>,
component::mapping::DistanceMultiMapping<defaulttype::Rigid3Types,defaulttype::Vec1Types>
> MultiDataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE( DistanceMultiMappingTest, MultiDataTypes );

// test case
TYPED_TEST( DistanceMultiMappingTest , twoParents )
{
    ASSERT_TRUE(this->test(2));
}

TYPED_TEST( DistanceMultiMappingTest , threeParents )
{
    ASSERT_TRUE(this->test(3));
}

TYPED_TEST( DistanceMultiMappingTest , fourParents )
{
    ASSERT_TRUE(this->test(4));
}

} // namespace
} // namespace sofa
