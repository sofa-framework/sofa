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
#include <gtest/gtest.h>

#include <sofa/helper/accessor.h>
#include <sofa/type/vector.h>

namespace sofa
{

TEST(WriteAccessor, PrimitiveTypes)
{
    float float_value { 12.f };
    sofa::helper::WriteAccessor float_accessor(float_value);
    EXPECT_FLOAT_EQ(float_accessor.ref(), 12.f);
    float_accessor.wref() = 14.f;
    EXPECT_FLOAT_EQ(float_accessor.ref(), 14.f);
    EXPECT_FLOAT_EQ(float_accessor, 14.f);
    EXPECT_FLOAT_EQ(float_value, 14.f);

    std::size_t size_t_value { 8 };
    sofa::helper::WriteAccessor size_t_accessor(size_t_value);
    EXPECT_EQ(size_t_accessor.ref(), 8);
    size_t_accessor.wref() = 9;
    EXPECT_EQ(size_t_accessor.ref(), 9);
    EXPECT_EQ(size_t_accessor, 9);
    EXPECT_EQ(size_t_value, 9);
}

TEST(WriteAccessor, VectorTypes)
{
    sofa::type::vector<float> vector { 0.f, 1.f, 2.f, 3.f, 4.f};
    sofa::helper::WriteAccessor accessor(vector);

    EXPECT_EQ(accessor.size(), vector.size());
    EXPECT_EQ(accessor.empty(), vector.empty());
    EXPECT_EQ(accessor.begin(), vector.begin());
    EXPECT_EQ(accessor.end(), vector.end());

    for(auto& v : accessor)
    {
        ++v;
    }

    EXPECT_FLOAT_EQ(vector[0], 1.f);
    EXPECT_FLOAT_EQ(vector[1], 2.f);
    EXPECT_FLOAT_EQ(vector[2], 3.f);
    EXPECT_FLOAT_EQ(vector[3], 4.f);
    EXPECT_FLOAT_EQ(vector[4], 5.f);

    EXPECT_FLOAT_EQ(accessor[0], 1.f);
    EXPECT_FLOAT_EQ(accessor[1], 2.f);
    EXPECT_FLOAT_EQ(accessor[2], 3.f);
    EXPECT_FLOAT_EQ(accessor[3], 4.f);
    EXPECT_FLOAT_EQ(accessor[4], 5.f);

    accessor[3] = 6.f;
    EXPECT_FLOAT_EQ(accessor[3], 6.f);
    EXPECT_FLOAT_EQ(vector[3], 6.f);
}


}
