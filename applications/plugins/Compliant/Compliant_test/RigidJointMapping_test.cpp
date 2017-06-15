/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
    typedef sofa::defaulttype::Vec<3,SReal> Vec3;
    
    Mapping* mapping;

    RigidJointMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {

        // we need to increase the error for avoiding numerical problem with quaternion
        this->errorMax = 1e10;
        this->deltaRange.second = this->errorMax*100;

        // parents
        typename self::InVecCoord xin(2);

        defaulttype::Quat q = defaulttype::Quat::fromEuler(M_PI_2,M_PI/8,M_PI_4);

        // finite difference method does not work well with large rotations
        // this seems mainly due to the fact that se3::log(q*dq) != se3::log(q) + se3::log(dq)
        // We only test the case when the parents share the same initial rotation
        xin[0].getOrientation() = q;
        xin[1].getOrientation() = q;

        se3::map(xin[0].getCenter()) << 0, 1, 1;
        se3::map(xin[1].getCenter()) << 0, 3, 0;

        typename self::OutVecCoord expected(1);  
        // the child is expressed in its first parent frame
        expected[0] = se3::product_log(se3::prod( se3::inv(xin[0]), xin[1])).getVAll();

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
