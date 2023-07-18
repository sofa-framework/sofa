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
#include <sofa/type/fixed_array.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(fixed_array, operatorLess)
{
    const sofa::type::fixed_array<sofa::Index, 2> edge1 { 0, 0};
    const sofa::type::fixed_array<sofa::Index, 2> edge2 { 1, 0};
    EXPECT_LT(edge1, edge2);
    EXPECT_GT(edge2, edge1);
}


TEST(fixed_array, structuredBindings)
{
    static constexpr sofa::type::fixed_array<std::size_t, 4> sofaArray { 8, -7, 4, -1};
    const auto& [a, b, c, d] = sofaArray;
    EXPECT_EQ(a, 8);
    EXPECT_EQ(b,-7);
    EXPECT_EQ(c, 4);
    EXPECT_EQ(d,-1);
}
}
