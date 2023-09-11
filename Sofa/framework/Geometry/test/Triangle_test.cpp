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

template <typename VecType>
class GeometryVec2DTriangle_test : public ::testing::Test
{
};
template <typename VecType>
class GeometryVec3DTriangle_test : public ::testing::Test
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

using SofaVec2Types = ::testing::Types <sofa::type::Vec < 2, float >, sofa::type::Vec < 2, double > >;
TYPED_TEST_SUITE(GeometryVec2DTriangle_test, SofaVec2Types);

using SofaVec3Types = ::testing::Types <sofa::type::Vec < 3, float >, sofa::type::Vec < 3, double > >;
TYPED_TEST_SUITE(GeometryVec3DTriangle_test, SofaVec3Types);



TYPED_TEST(Geometry2DTriangle_test, area)
{
    const TypeParam a{ -2.f, 3.f };
    const TypeParam b{ -3.f, -1.f };
    const TypeParam c{ 3.f, -2.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);

    EXPECT_FLOAT_EQ(float(testArea), 12.5f);
}

TYPED_TEST(Geometry3DTriangle_test, area)
{
    const TypeParam a{ -5.f, 5.f, -5.f };
    const TypeParam b{ 1.f, -6.f, 6.f };
    const TypeParam c{ 2.f, -3.f, 4.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(float(testArea), 19.306734f);
}

TYPED_TEST(Geometry2DTriangle_test, flat_area)
{
    const TypeParam a{ 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f };
    const TypeParam c{ 0.f, 1.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 0);
}

TYPED_TEST(Geometry3DTriangle_test, flat_area)
{
    const TypeParam a{ 0.f, 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f, 0.f };
    const TypeParam c{ 0.f, 1.f, 0.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_EQ(testArea, 0);
}


TYPED_TEST(Geometry3DTriangle_test, normal)
{
    // normal case
    const TypeParam a{ 0.f, 0.f, 0.f };
    const TypeParam b{ 0.f, 2.f, 0.f };
    const TypeParam c{ 0.f, 0.f, 2.f };
    
    auto normal = sofa::geometry::Triangle::normal(a, b, c);
    EXPECT_FLOAT_EQ(float(normal[0]), 4.f);
    EXPECT_FLOAT_EQ(float(normal[1]), 0.f);
    EXPECT_FLOAT_EQ(float(normal[2]), 0.f);

    // flat triangle case
    const TypeParam a2{ 0.f, 0.f, 0.f };
    const TypeParam b2{ 0.f, 2.f, 0.f };
    const TypeParam c2{ 0.f, 1.f, 0.f };
    
    normal = sofa::geometry::Triangle::normal(a2, b2, c2);
    EXPECT_FLOAT_EQ(float(normal[0]), 0.f);
    EXPECT_FLOAT_EQ(float(normal[1]), 0.f);
    EXPECT_FLOAT_EQ(float(normal[2]), 0.f);
}


TYPED_TEST(GeometryVec2DTriangle_test, isPointInTriangle)
{
    const TypeParam a{ 0., 0. };
    const TypeParam b{ 2., 0. };
    const TypeParam c{ 2., 2. };
    sofa::type::Vec3 bary;

    //// point inside
    TypeParam p0{ 1.5, 0.5 };
    auto res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(float(bary[0]), 0.25f);
    EXPECT_FLOAT_EQ(float(bary[1]), 0.5f);
    EXPECT_FLOAT_EQ(float(bary[2]), 0.25f);

    // barycenter
    p0 = { 4./3., 2./3. };
    const auto bVal = float(1.f / 3.f);
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_NEAR(bary[0], bVal, 1e-4);
    EXPECT_NEAR(bary[1], bVal, 1e-4);
    EXPECT_NEAR(bary[2], bVal, 1e-4);

    // on edge
    p0 = { 1., 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_NEAR(bary[0], 0.5f, 1e-4);
    EXPECT_NEAR(bary[1], 0.5f, 1e-4);
    EXPECT_NEAR(bary[2], 0.0f, 1e-4);

    // on corner
    p0 = { 2., 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(float(bary[0]), 0.0f);
    EXPECT_FLOAT_EQ(float(bary[1]), 1.0f);
    EXPECT_FLOAT_EQ(float(bary[2]), 0.0f);

    // False case
    p0 = { 4., 10. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_FALSE(res);
    EXPECT_NEAR(bary[0], -1.0f, 1e-4);
    EXPECT_NEAR(bary[1], -3.0f, 1e-4);
    EXPECT_NEAR(bary[2], 5.0f, 1e-4);


    // Special cases
    // flat triangle
    const TypeParam d{ 1., 0. };
    p0 = { 0.5, 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, d, bary);
    EXPECT_FALSE(res);
    EXPECT_EQ(bary[0], -1);
    EXPECT_EQ(bary[1], -1);
    EXPECT_EQ(bary[2], -1);

    //special False cases along edge
    p0 = { 3., 3. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_FALSE(res);
    EXPECT_NEAR(bary[0], -0.5f, 1e-4);
    EXPECT_NEAR(bary[1], 0.f, 1e-4);
    EXPECT_NEAR(bary[2], 1.5f, 1e-4);
}


TYPED_TEST(GeometryVec3DTriangle_test, isPointInTriangle)
{
    const TypeParam a{ 0., 0., 0. };
    const TypeParam b{ 2., 0., 2. };
    const TypeParam c{ 0., 2., 0. };
    
    sofa::type::Vec3 bary;

    // point inside
    TypeParam p0{ 0.5, 0.5, 0.5 };
    auto res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_NEAR(bary[0], 0.5f, 1e-4);
    EXPECT_NEAR(bary[1], 0.25f, 1e-4);
    EXPECT_NEAR(bary[2], 0.25f, 1e-4);

    // barycenter
    p0 = { 2. / 3., 2. / 3. , 2. / 3. };
    const auto bVal = float(1.f / 3.f);
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_NEAR(bary[0], bVal, 1e-4);
    EXPECT_NEAR(bary[1], bVal, 1e-4);
    EXPECT_NEAR(bary[2], bVal, 1e-4);

    // on edge
    p0 = { 1., 1., 1. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_NEAR(bary[0], 0.f, 1e-4);
    EXPECT_NEAR(bary[1], 0.5f, 1e-4);
    EXPECT_NEAR(bary[2], 0.5f, 1e-4);

    // on corner
    p0 = { 0., 2., 0. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_TRUE(res);
    EXPECT_EQ(bary[0], 0);
    EXPECT_EQ(bary[1], 0);
    EXPECT_EQ(bary[2], 1);

    // False cases
    // out of plan
    p0 = { 1., 0.2, 0.2 };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_FALSE(res);
    EXPECT_NEAR(bary[0], 0.69282, 1e-4);
    EXPECT_NEAR(bary[1], 0.360555, 1e-4);
    EXPECT_NEAR(bary[2], -0.0533754, 1e-4);

    // in plan but out of triangle
    p0 = { 2., 2., 2. };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, c, bary);
    EXPECT_FALSE(res);
    EXPECT_EQ(bary[0], 1);
    EXPECT_EQ(bary[1], 1);
    EXPECT_EQ(bary[2], -1);


    // Special cases
    // flat triangle
    const TypeParam d{ 1., 0., 1. };
    p0 = { 0.5, 0., 0.5 };
    res = sofa::geometry::Triangle::isPointInTriangle(p0, a, b, d, bary);
    EXPECT_FALSE(res);
    EXPECT_EQ(bary[0], -1);
    EXPECT_EQ(bary[1], -1);
    EXPECT_EQ(bary[2], -1);
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
