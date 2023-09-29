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

#include <sofa/geometry/Edge.h>

#include <sofa/type/fixed_array.h>
#include <array>
#include <sofa/type/Vec.h>

#include <gtest/gtest.h>


namespace sofa
{

TEST(GeometryEdge_test, squaredLength1f)
{
    const std::array<float, 1> a1{ 1.f };
    const std::array<float, 1> b1{ 10.f };

    const sofa::type::fixed_array<float, 1> a2{ 1.f };
    const sofa::type::fixed_array<float, 1> b2{ 10.f };

    const sofa::type::Vec1f a3{ 1.f };
    const sofa::type::Vec1f b3{ 10.f };

    const float expectedResult = 81.f;

    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a1, b1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a2, b2));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a3, b3));

    //special cases
    EXPECT_FLOAT_EQ(0.f, sofa::geometry::Edge::squaredLength(a1, a1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength((a3 * -1.f), (b3 * -1.f)));
}

TEST(GeometryEdge_test, squaredLength2f)
{
    const std::array<float, 2> a1{ 1.f, 1.f };
    const std::array<float, 2> b1{ 10.f, 10.f };
    const sofa::type::fixed_array<float, 2> a2{ 1.f, 1.f };
    const sofa::type::fixed_array<float, 2> b2{ 10.f, 10.f };
    const sofa::type::Vec2f a3{ 1.f, 1.f };
    const sofa::type::Vec2f b3{ 10.f, 10.f };

    const float expectedResult = 162.f;

    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a1, b1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a2, b2));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a3, b3));

    //special cases
    EXPECT_FLOAT_EQ(0.f, sofa::geometry::Edge::squaredLength(a1, a1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength((a3 * -1.f), (b3 * -1.f)));
}

TEST(GeometryEdge_test, squaredLength3f)
{
    const std::array<float, 3> a1{ 3.f, 2.f, 7.f };
    const std::array<float, 3> b1{ 8.f, 1.f, 9.f };
    const sofa::type::fixed_array<float, 3> a2{ 3.f, 2.f, 7.f };
    const sofa::type::fixed_array<float, 3> b2{ 8.f, 1.f, 9.f };
    const sofa::type::Vec3f a3{ 3.f, 2.f, 7.f };
    const sofa::type::Vec3f b3{ 8.f, 1.f, 9.f };

    const float expectedResult = 30.f;

    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a1, b1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a2, b2));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength(a3, b3));

    //special cases
    EXPECT_FLOAT_EQ(0.f, sofa::geometry::Edge::squaredLength(a1, a1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::squaredLength( (a3 * -1.f), (b3 * -1.f)));
}

TEST(GeometryEdge_test, length3f)
{
    const std::array<float, 3> a1{ 2.f, 1.f, 1.f };
    const std::array<float, 3> b1{ 5.f, 2.f, -1.f };
    const sofa::type::fixed_array<float, 3> a2{ 2.f, 1.f, 1.f };
    const sofa::type::fixed_array<float, 3> b2{ 5.f, 2.f, -1.f };
    const sofa::type::Vec3f a3{ 2.f, 1.f, 1.f };
    const sofa::type::Vec3f b3{ 5.f, 2.f, -1.f };

    const float expectedResult = 3.74165739f;

    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::length(a1, b1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::length(a2, b2));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::length(a3, b3));

    //special cases
    EXPECT_FLOAT_EQ(0.f, sofa::geometry::Edge::length(a1, a1));
    EXPECT_FLOAT_EQ(expectedResult, sofa::geometry::Edge::length((a3 * -1.f), (b3 * -1.f)));
}



TEST(GeometryEdge_test, getBarycentricCoordinates1f)
{
    const sofa::type::Vec1f a1{ 0.f };
    const sofa::type::Vec1f b1{ 2.f };
    
    const sofa::type::Vec1f p1{ 1.f };
    const auto res1 = sofa::geometry::Edge::getBarycentricCoordinates(p1, a1, b1);
    EXPECT_FLOAT_EQ(res1[0], 0.5f);
    EXPECT_FLOAT_EQ(res1[1], 0.5f);

    const sofa::type::Vec1f p2{ 0.25f };
    const auto res2 = sofa::geometry::Edge::getBarycentricCoordinates(p2, a1, b1);
    EXPECT_FLOAT_EQ(res2[0], 0.875f);
    EXPECT_FLOAT_EQ(res2[1], 0.125f);

    //special cases
    // Edge null
    const sofa::type::Vec1f c1{ 2.f };
    const auto res4 = sofa::geometry::Edge::getBarycentricCoordinates(p2, b1, c1);
    EXPECT_FLOAT_EQ(res4[0], 0.5f);
    EXPECT_FLOAT_EQ(res4[1], 0.5f);

    // Point on one Node
    const auto res5 = sofa::geometry::Edge::getBarycentricCoordinates(b1, a1, b1);
    EXPECT_FLOAT_EQ(res5[0], 0.0f);
    EXPECT_FLOAT_EQ(res5[1], 1.0f);

    // Point out of Edge
    const sofa::type::Vec1f p4{ -1.0f };
    const auto res6 = sofa::geometry::Edge::getBarycentricCoordinates(p4, a1, b1);
    EXPECT_FLOAT_EQ(res6[0], 1.5f);
    EXPECT_FLOAT_EQ(res6[1], 0.5f);
}

TEST(GeometryEdge_test, getBarycentricCoordinates2f)
{
    const sofa::type::Vec2f a1{ 0.f, 0.f };
    const sofa::type::Vec2f b1{ 2.f, 2.f };

    const sofa::type::Vec2f p1{ 1.f, 1.f };
    const auto res1 = sofa::geometry::Edge::getBarycentricCoordinates(p1, a1, b1);
    EXPECT_FLOAT_EQ(res1[0], 0.5f);
    EXPECT_FLOAT_EQ(res1[1], 0.5f);

    const sofa::type::Vec2f p2{ 0.25f, 0.25f};
    const auto res2 = sofa::geometry::Edge::getBarycentricCoordinates(p2, a1, b1);
    EXPECT_FLOAT_EQ(res2[0], 0.875f);
    EXPECT_FLOAT_EQ(res2[1], 0.125f);

    const sofa::type::Vec2f p3{ 1.0f, 0.25f };
    const auto res3 = sofa::geometry::Edge::getBarycentricCoordinates(p3, a1, b1);
    EXPECT_FLOAT_EQ(res3[0], 0.71260965);
    EXPECT_FLOAT_EQ(res3[1], 0.36443448);

    //special cases
    // Edge null
    const sofa::type::Vec2f c1{ 2.f, 2.f };
    const auto res4 = sofa::geometry::Edge::getBarycentricCoordinates(p2, b1, c1);
    EXPECT_FLOAT_EQ(res4[0], 0.5f);
    EXPECT_FLOAT_EQ(res4[1], 0.5f);

    // Point on one Node
    const auto res5 = sofa::geometry::Edge::getBarycentricCoordinates(b1, a1, b1);
    EXPECT_FLOAT_EQ(res5[0], 0.0f);
    EXPECT_FLOAT_EQ(res5[1], 1.0f);

    // Point out of Edge
    const sofa::type::Vec2f p4{ -1.0f, -1.0f };
    const auto res6 = sofa::geometry::Edge::getBarycentricCoordinates(p4, a1, b1);
    EXPECT_FLOAT_EQ(res6[0], 1.5f);
    EXPECT_FLOAT_EQ(res6[1], 0.5f);
}


TEST(GeometryEdge_test, getBarycentricCoordinates3f)
{
    const sofa::type::Vec3f a1{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b1{ 2.f, 2.f, 2.f };
    
    const sofa::type::Vec3f p1{ 1.f, 1.f, 1.f };
    const auto res1 = sofa::geometry::Edge::getBarycentricCoordinates(p1, a1, b1);
    EXPECT_FLOAT_EQ(res1[0], 0.5f);
    EXPECT_FLOAT_EQ(res1[1], 0.5f);

    const sofa::type::Vec3f p2{ 0.25f, 0.25f, 0.25f };
    const auto res2 = sofa::geometry::Edge::getBarycentricCoordinates(p2, a1, b1);
    EXPECT_FLOAT_EQ(res2[0], 0.875f);
    EXPECT_FLOAT_EQ(res2[1], 0.125f);

    const sofa::type::Vec3f p3{ 1.0f, 0.25f, 0.25f };
    const auto res3 = sofa::geometry::Edge::getBarycentricCoordinates(p3, a1, b1);
    EXPECT_FLOAT_EQ(res3[0], 0.77055174);
    EXPECT_FLOAT_EQ(res3[1], 0.3061862);
   
    //special cases
    // Edge null
    const sofa::type::Vec3f c1{ 2.f, 2.f, 2.f };
    const auto res4 = sofa::geometry::Edge::getBarycentricCoordinates(p2, b1, c1);
    EXPECT_FLOAT_EQ(res4[0], 0.5f);
    EXPECT_FLOAT_EQ(res4[1], 0.5f);

    // Point on one Node
    const auto res5 = sofa::geometry::Edge::getBarycentricCoordinates(b1, a1, b1);
    EXPECT_FLOAT_EQ(res5[0], 0.0f);
    EXPECT_FLOAT_EQ(res5[1], 1.0f);

    // Point out of Edge
    const sofa::type::Vec3f p4{ -1.0f, -1.0f, -1.0f };
    const auto res6 = sofa::geometry::Edge::getBarycentricCoordinates(p4, a1, b1);
    EXPECT_FLOAT_EQ(res6[0], 1.5f);
    EXPECT_FLOAT_EQ(res6[1], 0.5f);
}


TEST(GeometryEdge_test, isPointOnEdge2f)
{
    const sofa::type::Vec2f a1{ 0.f, 0.f };
    const sofa::type::Vec2f b1{ 2.f, 2.f };

    // True case
    const sofa::type::Vec2f p1{ 1.f, 1.f };
    const auto res1 = sofa::geometry::Edge::isPointOnEdge(p1, a1, b1);
    EXPECT_TRUE(res1);

    // True case
    const sofa::type::Vec2f p2{ 0.25f, 0.25f };
    const auto res2 = sofa::geometry::Edge::isPointOnEdge(p2, a1, b1);
    EXPECT_TRUE(res2);

    // False case
    const sofa::type::Vec2f p3{ 1.0f, 0.25f };
    const auto res3 = sofa::geometry::Edge::isPointOnEdge(p3, a1, b1);
    EXPECT_FALSE(res3);


    //special cases
    // Edge null
    const sofa::type::Vec2f c1{ 2.f, 2.f };
    const auto res4 = sofa::geometry::Edge::isPointOnEdge(p1, b1, c1);
    EXPECT_FALSE(res4);

    // Point on one Node
    const auto res5 = sofa::geometry::Edge::isPointOnEdge(b1, a1, b1);
    EXPECT_TRUE(res5);

    // Point out of Edge borders
    const sofa::type::Vec2f p4{ -1.0f, -1.0f };
    const sofa::type::Vec2f p5{ 2.5f, 2.5f };
    const auto res6 = sofa::geometry::Edge::isPointOnEdge(p4, a1, b1);
    const auto res7 = sofa::geometry::Edge::isPointOnEdge(p5, a1, b1);

    EXPECT_FALSE(res6);
    EXPECT_FALSE(res7);
}


TEST(GeometryEdge_test, isPointOnEdge3f)
{
    const sofa::type::Vec3f a1{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b1{ 2.f, 2.f, 2.f };

    // True case
    const sofa::type::Vec3f p1{ 1.f, 1.f, 1.f };
    const auto res1 = sofa::geometry::Edge::isPointOnEdge(p1, a1, b1);
    EXPECT_TRUE(res1);

    // True case
    const sofa::type::Vec3f p2{ 0.25f, 0.25f, 0.25f };
    const auto res2 = sofa::geometry::Edge::isPointOnEdge(p2, a1, b1);
    EXPECT_TRUE(res2);

    // False case
    const sofa::type::Vec3f p3{ 1.0f, 0.25f, 0.25f };
    const auto res3 = sofa::geometry::Edge::isPointOnEdge(p3, a1, b1);
    EXPECT_FALSE(res3);


    //special cases
    // Edge null
    const sofa::type::Vec3f c1{ 2.f, 2.f, 2.f };
    const auto res4 = sofa::geometry::Edge::isPointOnEdge(p1, b1, c1);
    EXPECT_FALSE(res4);

    // Point on one Node
    const auto res5 = sofa::geometry::Edge::isPointOnEdge(b1, a1, b1);
    EXPECT_TRUE(res5);

    // Point out of Edge borders
    const sofa::type::Vec3f p4{ -1.0f, -1.0f, -1.0f };
    const sofa::type::Vec3f p5{ 2.5f, 2.5f, 2.5f };
    const auto res6 = sofa::geometry::Edge::isPointOnEdge(p4, a1, b1);
    const auto res7 = sofa::geometry::Edge::isPointOnEdge(p5, a1, b1);

    EXPECT_FALSE(res6);
    EXPECT_FALSE(res7);
}

TEST(GeometryEdge_test, intersectionWithEdge2f)
{
    const sofa::type::Vec2f e01{ 0.f, 0.f };
    const sofa::type::Vec2f e02{ 2.f, 2.f };

    const sofa::type::Vec2f e11{ 0.f, 2.f };
    sofa::type::Vec2f e12{ 2.f, 0.f };

    sofa::type::Vec2f inter{ 0.f, 0.f };

    // basic cases
    // intersection in the middle
    auto res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(inter[0], 1.0f);
    EXPECT_FLOAT_EQ(inter[1], 1.0f);

    // intersection on a node
    e12 = { 2.f, 2.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(inter[0], 2.0f);
    EXPECT_FLOAT_EQ(inter[1], 2.0f);

    // no intersection
    e12 = { -1.f, -1.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());

    // colinear
    e12 = { 2.f, 4.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());

    // on the same line but no overlapping    
    sofa::type::Vec2f e13{ 2.001f, 2.001f };
    sofa::type::Vec2f e14{ 3.f, 3.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e13, e14, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());

    // on the same line and overlapping    
    e13 = { 1.5f, 1.5f };
    e14 = { 3.f, 3.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e13, e14, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());
}



TEST(GeometryEdge_test, intersectionWithEdge3f)
{
    const sofa::type::Vec3f e01{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f e02{ 2.f, 2.f, 2.f };

    const sofa::type::Vec3f e11{ 0.f, 0.f, 2.f };
    sofa::type::Vec3f e12{ 2.f, 2.f, 0.f };

    sofa::type::Vec3f inter{ 0.f, 0.f, 0.f };

    // basic cases
    // intersection in the middle
    auto res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(inter[0], 1.0f);
    EXPECT_FLOAT_EQ(inter[1], 1.0f);
    EXPECT_FLOAT_EQ(inter[2], 1.0f);

    // intersection on a node
    e12 = { 2.f, 2.f, 2.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_TRUE(res);
    EXPECT_FLOAT_EQ(inter[0], 2.0f);
    EXPECT_FLOAT_EQ(inter[1], 2.0f);
    EXPECT_FLOAT_EQ(inter[2], 2.0f);

    // no intersection
    e12 = { -1.f, -1.f, -1.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[2], std::numeric_limits<float>::min());

    // colinear
    e12 = { 2.f, 2.f, 4.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e11, e12, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[2], std::numeric_limits<float>::min());

    // on the same line but no overlapping    
    sofa::type::Vec3f e13{ 2.001f, 2.001f, 2.001f };
    sofa::type::Vec3f e14{ 3.f, 3.f, 3.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e13, e14, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[2], std::numeric_limits<float>::min());

    // on the same line and overlapping    
    e13 = { 1.5f, 1.5f, 1.5f };
    e14 = { 3.f, 3.f, 3.f };
    res = sofa::geometry::Edge::intersectionWithEdge(e01, e02, e13, e14, inter);
    EXPECT_FALSE(res);
    EXPECT_FLOAT_EQ(inter[0], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[1], std::numeric_limits<float>::min());
    EXPECT_FLOAT_EQ(inter[2], std::numeric_limits<float>::min());
}

TEST(GeometryEdge_test, intersectionWithPlane3f)
{
    const sofa::type::Vec3f a1{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b1{ 2.f, 2.f, 2.f };

    const sofa::type::Vec3f planeP{ 0.f, 1.f, 0.f };
    const sofa::type::Vec3f planeN1{ 0.f, 1.f, 0.f };
    sofa::type::Vec3f inter{ 0.f, 0.f, 0.f };

    // basic cases
    bool res = sofa::geometry::Edge::intersectionWithPlane(a1, b1, planeP, planeN1, inter);
    EXPECT_EQ(res, true);
    EXPECT_FLOAT_EQ(inter[0], 1.0f);
    EXPECT_FLOAT_EQ(inter[1], 1.0f);
    EXPECT_FLOAT_EQ(inter[2], 1.0f);

    const sofa::type::Vec3f planeN2{ 1.f, 2.f, 1.f };
    bool res2 = sofa::geometry::Edge::intersectionWithPlane(a1, b1, planeP, planeN2, inter);
    EXPECT_EQ(res2, true);
    EXPECT_FLOAT_EQ(inter[0], 0.5f);
    EXPECT_FLOAT_EQ(inter[1], 0.5f);
    EXPECT_FLOAT_EQ(inter[2], 0.5f);

    // border case: plan - Edge intersection on Edge node
    const sofa::type::Vec3f planeN3{ 1.f, 0.f, 0.f };
    bool res3 = sofa::geometry::Edge::intersectionWithPlane(a1, b1, planeP, planeN3, inter);
    EXPECT_EQ(res3, true);
    EXPECT_FLOAT_EQ(inter[0], 0.0f);
    EXPECT_FLOAT_EQ(inter[1], 0.0f);
    EXPECT_FLOAT_EQ(inter[2], 0.0f);

    // negative case: plan - Edge has no intersection
    const sofa::type::Vec3f planeN4{ -1.f, 2.f, -1.f };
    bool res4 = sofa::geometry::Edge::intersectionWithPlane(a1, b1, planeP, planeN4, inter);
    EXPECT_EQ(res4, false);
    EXPECT_FLOAT_EQ(inter[0], 0.0f);
    EXPECT_FLOAT_EQ(inter[1], 0.0f);
    EXPECT_FLOAT_EQ(inter[2], 0.0f);

    // failing case: Edge length == 0
    bool res5 = sofa::geometry::Edge::intersectionWithPlane(a1, a1, planeP, planeN1, inter);
    EXPECT_EQ(res5, false);
    EXPECT_FLOAT_EQ(inter[0], 0.0f);
    EXPECT_FLOAT_EQ(inter[1], 0.0f);
    EXPECT_FLOAT_EQ(inter[2], 0.0f);
}

}// namespace sofa
