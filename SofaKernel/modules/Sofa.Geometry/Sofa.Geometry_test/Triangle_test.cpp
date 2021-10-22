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
#include <array>

#include <gtest/gtest.h>

namespace sofa
{

TEST(GeometryTriangle_test, area2f_stdarray)
{
    const std::array<float, 2> a{ -2.f, 3.f };
    const std::array<float, 2> b{ -3.f, -1.f };
    const std::array<float, 2> c{ 3.f, -2.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 12.5f);
}
TEST(GeometryTriangle_test, area3f_stdarray)
{
    const std::array<float, 3> a{ -5.f, 5.f, -5.f };
    const std::array<float, 3> b{ 1.f, -6.f, 6.f };
    const std::array<float, 3> c{ 2.f, -3.f, 4.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 19.306734f);
}
TEST(GeometryTriangle_test, flat_area2f_stdarray)
{
    const std::array<float, 2> a{ 0.f, 0.f };
    const std::array<float, 2> b{ 0.f, 2.f };
    const std::array<float, 2> c{ 0.f, 1.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 0.f);
}
TEST(GeometryTriangle_test, flat_area3f_stdarray)
{
    const std::array<float, 3> a{ 0.f, 0.f, 0.f };
    const std::array<float, 3> b{ 0.f, 2.f, 0.f };
    const std::array<float, 3> c{ 0.f, 1.f, 0.f };

    const auto testArea = sofa::geometry::Triangle::area(a, b, c);
    EXPECT_FLOAT_EQ(testArea, 0.f);
}

}// namespace sofa
