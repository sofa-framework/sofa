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


TYPED_TEST(Geometry3DTriangle_test, normal)
{
    // normal case
    const TypeParam a{ 0.f, 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f, 0.f };
    const TypeParam c{ 0.f, 0.f, 2.f };

    auto normal = sofa::geometry::Triangle::normal(a, b, c);
    EXPECT_FLOAT_EQ(normal[0], 4.f);
    EXPECT_FLOAT_EQ(normal[1], 0.f);
    EXPECT_FLOAT_EQ(normal[2], 0.f);

    // flat triangle case
    const TypeParam a2{ 0.f, 0.f, 0.f };
    const TypeParam b2{ 0.f, 2.f, 0.f };
    const TypeParam c2{ 0.f, 1.f, 0.f };
    
    normal = sofa::geometry::Triangle::normal(a2, b2, c2);
    EXPECT_FLOAT_EQ(normal[0], 0.f);
    EXPECT_FLOAT_EQ(normal[1], 0.f);
    EXPECT_FLOAT_EQ(normal[2], 0.f);
}


TEST(GeometryTriangle_test, isPointInTriangle2)
{
    const sofa::type::Vec2 a{ 0., 0. };
    const sofa::type::Vec2 b{ 2., 0. };
    const sofa::type::Vec2 c{ 2., 2. };
    sofa::type::Vec3 bary;

    sofa::type::Vec2 p0{ 1., 0.5 };
    auto res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, c);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(bary[0], 0.5f);
    EXPECT_FLOAT_EQ(bary[1], 0.25f);
    EXPECT_FLOAT_EQ(bary[2], 0.25f);

    p0 = { 1., 1. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, c);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(bary[0], 1.0f);
    EXPECT_FLOAT_EQ(bary[1], 0.5f);
    EXPECT_FLOAT_EQ(bary[2], 0.5f);

    p0 = { 2., 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, c);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(bary[0], 1.0f);
    EXPECT_FLOAT_EQ(bary[1], 0.5f);
    EXPECT_FLOAT_EQ(bary[2], 0.5f);


    p0 = { 0., 4. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, c);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(bary[0], 1.0f);
    EXPECT_FLOAT_EQ(bary[1], 0.5f);
    EXPECT_FLOAT_EQ(bary[2], 0.5f);

    p0 = { 2.1, 2.1 };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, c);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(bary[0], 1.0f);
    EXPECT_FLOAT_EQ(bary[1], 0.5f);
    EXPECT_FLOAT_EQ(bary[2], 0.5f);


    const sofa::type::Vec2 d{ 3., 0. };
    p0 = { 2.5, 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, d);
    bary = sofa::geometry::Triangle::pointBaryCoefs(p0, a, b, d);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(bary[0], 1.0f);
    EXPECT_FLOAT_EQ(bary[1], 0.5f);
    EXPECT_FLOAT_EQ(bary[2], 0.5f);
}


TEST(GeometryTriangle_test, rayIntersectionVec3)
{
    const sofa::type::Vec3 a{ 0., 3., 0. };
    const sofa::type::Vec3 b{ -0.5, 3., -1. };
    const sofa::type::Vec3 c{ 0.5, 3., -1. };
    const sofa::type::Vec3 origin{ 0. , 2., -1. };
    sofa::type::Vec3 direction{ 0., 1., 0. };

    SReal t{}, u{}, v{};
    EXPECT_TRUE(sofa::geometry::Triangle::rayIntersection(a, b, c, origin, direction, t, u, v));
    EXPECT_FLOAT_EQ(t, 1.0f);
    EXPECT_FLOAT_EQ(u, 0.5f);
    EXPECT_FLOAT_EQ(v, 0.5f);

    direction = { 0., 1., 2. };
    EXPECT_FALSE(sofa::geometry::Triangle::rayIntersection(a, b, c, origin, direction));
}

}// namespace sofa
