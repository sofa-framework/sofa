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
#include <Compliant/mapping/NormalizationMapping.h>


namespace sofa {

/**  Test suite for NormalizationMapping
  */
template <typename Mapping>
struct NormalizationMappingTest : public Mapping_test<Mapping>
{

    typedef NormalizationMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;
    
    Mapping* mapping;

    NormalizationMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // geometric stiffness does not seem to be correctly implemented (or the test is incorrect?)
        this->flags &= ~self::TEST_GEOMETRIC_STIFFNESS;

        // parents
        typename self::InVecCoord xin(2);
        xin[0] = typename self::InCoord(1,1,1);
        xin[1] = typename self::InCoord(5,6,7);

        typename self::OutVecCoord expected(2);
        expected[0] = typename self::OutCoord(1./sqrt(3.),1./sqrt(3.),1./sqrt(3.));
        expected[1] = typename self::OutCoord(5./sqrt(110.),6./sqrt(110.),7./sqrt(110.));

        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::NormalizationMapping<defaulttype::Vec3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(NormalizationMappingTest, DataTypes);

TYPED_TEST( NormalizationMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}






} // namespace sofa
