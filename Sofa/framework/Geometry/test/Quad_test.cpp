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

#include <sofa/geometry/Quad.h>

#include <sofa/type/fixed_array.h>
#include <array>

#include <gtest/gtest.h>

namespace sofa
{

TEST(GeometryQuad_test, square_area3f_stdarray)
{
    const std::array<float, 3> a{ -2.f, -2.f, 1.f };
    const std::array<float, 3> b{ 6.f, -2.f, 1.f };
    const std::array<float, 3> c{ 6.f, 6.f, 1.f };
    const std::array<float, 3> d{ -2.f, 6.f, 1.f };

    const auto testArea = sofa::geometry::Quad::area(a, b, c, d);
    const auto expectedArea = 8.f * 8.f;
    EXPECT_FLOAT_EQ(testArea, expectedArea);
}

TEST(GeometryQuad_test, quad_area3f_stdarray)
{
    const std::array<float, 3> a{ 5.f, 0.f, 0.f };
    const std::array<float, 3> b{ 0.f, 0.f, 2.f };
    const std::array<float, 3> c{ 0.f, 4.f, 0.f };
    const std::array<float, 3> d{ 5.f, 6.f, -3.f };

    const auto testArea = sofa::geometry::Quad::area(a, b, c, d);
    const auto expectedArea = 29.685856f;
    EXPECT_FLOAT_EQ(testArea, expectedArea);
}

}// namespace sofa
