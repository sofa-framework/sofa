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
#include <Compliant/mapping/SafeDistanceMapping.h>


namespace sofa {

/**  Test suite for SafeDistanceMapping
  */
template <typename Mapping>
struct SafeDistanceMappingTest : public Mapping_test<Mapping>
{

    typedef SafeDistanceMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;
    
    Mapping* mapping;

    SafeDistanceMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // we need to increase the error, the mapping is too much non-linear
        // and the finite differences are too different from the analytic Jacobian
        this->errorMax *= 300;

        // mapping parameters
        typename Mapping::pairs_type pairs(3);
        pairs[0][0] = 0; pairs[0][1] = 1;
        pairs[1][0] = 0; pairs[1][1] = 2;
        pairs[2][0] = 0; pairs[2][1] = 2;
        mapping->d_pairs.setValue(pairs);

        helper::vector<SReal> restLengths(3);
        restLengths[0] = 0;
        restLengths[1] = 0;
        restLengths[2] = 1;
        mapping->d_restLengths.setValue(restLengths);

        mapping->d_geometricStiffness.setValue(1); // exact


        // parents
        typename self::InVecCoord xin(3);
        xin[0] = typename self::InCoord(0,0,0);
        xin[1] = typename self::InCoord(325,23,-54);
        xin[2] = typename self::InCoord(1e-5,-1e-5,1e-7);

        typename self::OutVecCoord expected(5);
        expected[0] = typename self::OutCoord(xin[1].norm());
        expected[1] = typename self::OutCoord(xin[2][0]);
        expected[2] = typename self::OutCoord(xin[2][1]);
        expected[3] = typename self::OutCoord(xin[2][2]);
        expected[4] = typename self::OutCoord(xin[2].norm()-restLengths[2]);

        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::SafeDistanceMapping<defaulttype::Vec3Types, defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(SafeDistanceMappingTest, DataTypes);

TYPED_TEST( SafeDistanceMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}




///////////////////////////




/**  Test suite for SafeDistanceFromTargetMapping
  */
template <typename Mapping>
struct SafeDistanceFromTargetMappingTest : public Mapping_test<Mapping>
{
    typedef SafeDistanceFromTargetMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;

    Mapping* mapping;

    SafeDistanceFromTargetMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test_differencefailsafe()
    {
        // we need to increase the error, the mapping is too much non-linear
        // and the finite differences are too different from the analytic Jacobian
        this->errorMax *= 300;

        // mapping parameters
        helper::vector<unsigned> indices(3);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 1;
        mapping->d_indices.setValue(indices);

        typename self::InVecCoord targets(3);
        targets[0] = typename self::InCoord(0,0,0);
        targets[1] = typename self::InCoord(0,0,0);
        targets[2] = typename self::InCoord(0,0,0);
        mapping->d_targetPositions.setValue(targets);

        helper::vector<SReal> restLengths(3);
        restLengths[0] = 0;
        restLengths[1] = 0;
        restLengths[2] = 1;
        mapping->d_restLengths.setValue(restLengths);

        mapping->d_geometricStiffness.setValue(1); // exact


        // parents
        typename self::InVecCoord xin(2);
        xin[0] = typename self::InCoord(325,23,-54);
        xin[1] = typename self::InCoord(1e-5,-1e-5,1e-7);

        typename self::OutVecCoord expected(5);
        expected[0] = typename self::OutCoord(xin[0].norm());
        expected[1] = typename self::OutCoord(xin[1][0]);
        expected[2] = typename self::OutCoord(xin[1][1]);
        expected[3] = typename self::OutCoord(xin[1][2]);
        expected[4] = typename self::OutCoord(xin[1].norm()-restLengths[2]);

        return this->runTest(xin, expected);
    }

    bool test_givendirections()
    {
        // we need to increase the error, the mapping is too much non-linear
        // and the finite differences are too different from the analytic Jacobian
        this->errorMax *= 300;

        // mapping parameters
        helper::vector<unsigned> indices(3);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 1;
        mapping->d_indices.setValue(indices);

        typename self::InVecCoord targets(3);
        targets[0] = typename self::InCoord(0,0,0);
        targets[1] = typename self::InCoord(0,0,0);
        targets[2] = typename self::InCoord(0,0,0);
        mapping->d_targetPositions.setValue(targets);

        helper::vector<SReal> restLengths(3);
        restLengths[0] = 0;
        restLengths[1] = 0;
        restLengths[2] = 1;
        mapping->d_restLengths.setValue(restLengths);

        helper::vector<defaulttype::Vector3> directions(3);
        directions[0] = defaulttype::Vector3(325,23,-54);
        directions[1] = defaulttype::Vector3(1,0,0);
        directions[2] = defaulttype::Vector3(1,0,0);
        mapping->d_directions.setValue(directions);

        mapping->d_geometricStiffness.setValue(1); // exact


        // parents
        typename self::InVecCoord xin(2);
        xin[0] = typename self::InCoord(325,23,-54);
        xin[1] = typename self::InCoord(1e-5,-1e-5,1e-7);

        typename self::OutVecCoord expected(3);
        expected[0] = typename self::OutCoord(xin[0].norm());
        expected[1] = typename self::OutCoord(xin[1].norm());
        expected[2] = typename self::OutCoord(xin[1].norm()-restLengths[2]);

        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::SafeDistanceFromTargetMapping<defaulttype::Vec3Types, defaulttype::Vec1Types>
> DataTypes2; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(SafeDistanceFromTargetMappingTest, DataTypes2);

TYPED_TEST( SafeDistanceFromTargetMappingTest, test_differencefailsafe )
{
    ASSERT_TRUE( this->test_differencefailsafe() );
}

TYPED_TEST( SafeDistanceFromTargetMappingTest, test_givendirections )
{
    ASSERT_TRUE( this->test_givendirections() );
}




} // namespace sofa
