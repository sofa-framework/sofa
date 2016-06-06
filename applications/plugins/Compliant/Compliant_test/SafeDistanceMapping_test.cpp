/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
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
        this->errorMax *= 200;

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



} // namespace sofa
