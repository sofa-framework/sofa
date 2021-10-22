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

}// namespace sofa
