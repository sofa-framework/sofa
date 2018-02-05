/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <SofaTest/Mapping_test.h>
#include <SofaTest/MultiMapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Compliant/mapping/DotProductMapping.h>


namespace sofa {

/**  Test suite for DotProductMapping
  */
template <typename Mapping>
struct DotProductMappingTest : public Mapping_test<Mapping>
{

    typedef DotProductMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;
    
    Mapping* mapping;

    DotProductMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // we need to increase the error for avoiding numerical problem
        this->errorMax *= 1000;
        this->deltaRange.first = this->errorMax*100;
        this->deltaRange.second = this->errorMax*1000;

        // parents
        typename self::InVecCoord xin(4);

        xin[0] = typename self::InCoord(1,1,1);
        xin[1] = typename self::InCoord(5,6,7);
        xin[2] = typename self::InCoord(12,-3,6);
        xin[3] = typename self::InCoord(8,-2,-4);

        typename self::OutVecCoord expected(3);
        expected[0] = typename self::OutCoord(18);
        expected[1] = typename self::OutCoord(78);
        expected[2] = typename self::OutCoord(2);

        // mapping parameters
        typename Mapping::pairs_type pairs(3);
        pairs[0][0] = 0; pairs[0][1] = 1;
        pairs[1][0] = 2; pairs[1][1] = 3;
        pairs[2][0] = 0; pairs[2][1] = 3;
        mapping->pairs.setValue(pairs);

        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::DotProductMapping<defaulttype::Vec3Types, defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(DotProductMappingTest, DataTypes);

TYPED_TEST( DotProductMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}



////////////////////////



/**  Test suite for DotProductMultiMapping
  */
template <typename Mapping>
struct DotProductMultiMappingTest : public MultiMapping_test<Mapping>
{

    typedef DotProductMultiMappingTest self;
    typedef MultiMapping_test<Mapping> base;


    Mapping* mapping;

    DotProductMultiMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // we need to increase the error for avoiding numerical problem
        this->errorMax *= 1000;
        this->deltaRange.first = this->errorMax*100;
        this->deltaRange.second = this->errorMax*1000;

        const int NP = 2;
        this->setupScene(NP); // NP parents, 1 child

        // parent positions
        helper::vector< typename self::InVecCoord > incoords(NP);
        for( int i=0; i<NP; i++ )
        {
            incoords[i].resize(2);
        }
        self::In::set( incoords[0][0], 1,1,1 );
        self::In::set( incoords[0][1], 65,3,-51 );
        self::In::set( incoords[1][0], 23,35,-4 );
        self::In::set( incoords[1][1], -100,100,20 );

        // expected child positions
        typename self::OutVecCoord outcoords(2);
        outcoords[0][0] = 54;
        outcoords[1][0] = -7220;

        return this->runTest(incoords,outcoords);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
typedef Types<
    component::mapping::DotProductMultiMapping<defaulttype::Vec3Types, defaulttype::Vec1Types>
> DataTypes2; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(DotProductMultiMappingTest, DataTypes2);

TYPED_TEST( DotProductMultiMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

/////////////////////



/**  Test suite for DotProductFromTargetMapping
  */
template <typename Mapping>
struct DotProductFromTargetMappingTest : public Mapping_test<Mapping>
{

    typedef DotProductFromTargetMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;

    Mapping* mapping;

    DotProductFromTargetMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // we need to increase the error for avoiding numerical problem
        this->errorMax *= 1000;
        this->deltaRange.first = this->errorMax*100;
        this->deltaRange.second = this->errorMax*1000;

        // parents
        typename self::InVecCoord xin(4);
        xin[0] = typename self::InCoord(1,1,1);
        xin[1] = typename self::InCoord(5,6,7);
        xin[2] = typename self::InCoord(12,-3,6);
        xin[3] = typename self::InCoord(8,-2,-4);

        typename self::OutVecCoord expected(3);
        expected[0] = typename self::OutCoord(20);
        expected[1] = typename self::OutCoord(18);
        expected[2] = typename self::OutCoord(15);

        // mapping parameters
        helper::vector<unsigned> indices(3);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 2;
        mapping->d_indices.setValue(indices);

        typename self::InVecCoord targets(2);
        targets[0] = typename self::InCoord(10,-20,30);
        targets[1] = typename self::InCoord(1,1,1);
        mapping->d_targets.setValue(targets);


        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
typedef Types<
    component::mapping::DotProductFromTargetMapping<defaulttype::Vec3Types, defaulttype::Vec1Types>
> DataTypes3; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(DotProductFromTargetMappingTest, DataTypes3);

TYPED_TEST( DotProductFromTargetMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}



} // namespace sofa
