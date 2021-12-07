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

#include <sofa/geometry/Triangle.h>

#include <sofa/type/fixed_array.h>
#include <sofa/type/Vec.h>
#include <array>

#include <gtest/gtest.h>

namespace sofa
{
template <typename VecType>
class Geometry2DTriangle_test : public ::testing::Test
{
};
template <typename VecType>
class Geometry3DTriangle_test : public ::testing::Test
{
};

using Vec2Types = ::testing::Types <
    std::array<float, 2>, sofa::type::fixed_array<float, 2>, sofa::type::Vec < 2, float >,
    std::array<double, 2>, sofa::type::fixed_array<double, 2>, sofa::type::Vec < 2, double >>;
TYPED_TEST_SUITE(Geometry2DTriangle_test, Vec2Types);

using Vec3Types = ::testing::Types <
    std::array<float, 3>, sofa::type::fixed_array<float, 3>, sofa::type::Vec < 3, float >,
    std::array<double, 3>, sofa::type::fixed_array<double, 3>, sofa::type::Vec < 3, double >>;
TYPED_TEST_SUITE(Geometry3DTriangle_test, Vec3Types);


TYPED_TEST(Geometry2DTriangle_test, area)
{
    const TypeParam a{ -2.f, 3.f };
    const TypeParam b{ -3.f, -1.f };
    const TypeParam c{ 3.f, -2.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);

    EXPECT_FLOAT_EQ(testArea, 12.5f);
}

TYPED_TEST(Geometry3DTriangle_test, area)
{
    const TypeParam a{ -5.f, 5.f, -5.f };
    const TypeParam b{ 1.f, -6.f, 6.f };
    const TypeParam c{ 2.f, -3.f, 4.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 19.306734f);
}

TYPED_TEST(Geometry2DTriangle_test, flat_area)
{
    const TypeParam a{ 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f };
    const TypeParam c{ 0.f, 1.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 0.f);
}

TYPED_TEST(Geometry3DTriangle_test, flat_area)
{
    const TypeParam a{ 0.f, 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f, 0.f };
    const TypeParam c{ 0.f, 1.f, 0.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 0.f);
}


}// namespace sofa
