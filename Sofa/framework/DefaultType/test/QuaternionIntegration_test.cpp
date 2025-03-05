/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa {
namespace {

// checks that a 1 second rotation at 1 rad.s^{-1} rotates by 1 rad.
template<class Rigid3Types>
struct QuaternionIntegrationTest : NumericTest< typename Rigid3Types::Real > {

    using data_types = Rigid3Types;

    typename data_types::Coord coord;
    typename data_types::Deriv deriv;

    typename data_types::Real dt;

    QuaternionIntegrationTest()
        : dt(1) {

        deriv = data_types::randomDeriv(0, M_PI / 2);

        // time integration
        coord += deriv * dt;
    }


    void test_quaternion_angle() const {
        using real = typename data_types::Real;
        type::Vec<3, real> axis; real angle;
        coord.getOrientation().quatToAxis(axis, angle);

        const real expected_angle = dt * deriv.getVOrientation().norm();

        // child coordinates given directly in parent frame
        ASSERT_TRUE(this->isSmall(expected_angle - angle, 10));
    }

};


// Define the list of types to instantiate. We do not necessarily need to test all combinations.
using ::testing::Types;
typedef Types<defaulttype::Rigid3Types> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(QuaternionIntegrationTest, DataTypes);

// first test case
TYPED_TEST(QuaternionIntegrationTest, quaternion_angle) {
    EXPECT_MSG_NOEMIT(Error);
    this->test_quaternion_angle();
}

}
}
