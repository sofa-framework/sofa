/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// #include "stdafx.h"

#include <SofaTest/Mapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Compliant/mapping/RigidJointMapping.h>

#include <Compliant/utils/se3.h>
#include <Compliant/utils/edit.h>


namespace sofa {

/**  Test suite for RigidJointMapping
  */
template <typename Mapping>
struct RigidJointMappingTest : public Mapping_test<Mapping>
{

    typedef RigidJointMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef SE3< typename self::Real > se3;
    
    Mapping* mapping;

    RigidJointMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
//        this->errorMax *= 10;
    }

    bool test()
    {

        // parent
        typename self::InVecCoord xin(2);
        typename se3::vec3 v;

        v << M_PI / 2, 0, 0;
        xin[0].getOrientation() = se3::coord( se3::exp(v) );

        se3::map(xin[0].getCenter()) << 0, 2, 0;
        se3::map(xin[1].getCenter()) << 0, 3, 0;

        typename self::OutVecCoord expected(1);     
        expected[0].set(1,0,0,0,0,-M_PI / 2);

        // mapping parameters
        typename Mapping::pairs_type pairs(1);
        pairs[0][0] = 0;
        pairs[0][1] = 1;

        mapping->pairs.setValue(pairs);
        this->flags = base::TEST_getJs;
        
        return this->runTest(xin, expected);
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::RigidJointMapping<defaulttype::Rigid3Types,
                                                   defaulttype::Vec6Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(RigidJointMappingTest, DataTypes);

TYPED_TEST( RigidJointMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

} // namespace sofa
