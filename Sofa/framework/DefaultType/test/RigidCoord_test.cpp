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
#include <gtest/gtest.h>
#include <sofa/defaulttype/RigidCoord.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{

using defaulttype::RigidCoord;

TEST(RigidCoord3Iterator, begin)
{
    constexpr RigidCoord<3, SReal> r;
    constexpr auto it = r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidCoord3Iterator, begin_begin)
{
    constexpr RigidCoord<3, SReal> r;
    constexpr auto it1 = r.begin();
    constexpr auto it2 = r.begin();
    EXPECT_EQ(it1, it2);
}

TEST(RigidCoord3Iterator, begin_end)
{
    constexpr RigidCoord<3, SReal> r;
    constexpr auto begin = r.begin();
    constexpr auto end = r.end();
    EXPECT_NE(begin, end);
}

TEST(RigidCoord3Iterator, pre_increment)
{
    constexpr RigidCoord<3, SReal> r(type::Vec<3,SReal>{0, 1, 2}, type::Quat<SReal>{3, 4, 5, 6});
    constexpr auto it = ++r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 1_sreal);
}

TEST(RigidCoord3Iterator, post_increment)
{
    constexpr RigidCoord<3, SReal> r(type::Vec<3,SReal>{0, 1, 2}, type::Quat<SReal>{3, 4, 5, 6});
    constexpr auto it = r.begin()++;
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidCoord3Iterator, for_loop)
{
    constexpr RigidCoord<3, SReal> coord(type::Vec<3,SReal>{0, 1, 2}, type::Quat<SReal>{3, 4, 5, 6});

    sofa::Size i {};
    for (const auto& r : coord)
    {
        EXPECT_FLOATINGPOINT_EQ(r, static_cast<SReal>(i++));
    }
}




TEST(RigidCoord2Iterator, begin)
{
    constexpr RigidCoord<2, SReal> r;
    constexpr auto it = r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidCoord2Iterator, begin_begin)
{
    constexpr RigidCoord<2, SReal> r;
    constexpr auto it1 = r.begin();
    constexpr auto it2 = r.begin();
    EXPECT_EQ(it1, it2);
}

TEST(RigidCoord2Iterator, begin_end)
{
    constexpr RigidCoord<2, SReal> r;
    constexpr auto begin = r.begin();
    constexpr auto end = r.end();
    EXPECT_NE(begin, end);
}

TEST(RigidCoord2Iterator, pre_increment)
{
    constexpr RigidCoord<2, SReal> r(type::Vec<2,SReal>{0, 1}, 3_sreal);
    constexpr auto it = ++r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 1_sreal);
}

TEST(RigidCoord2Iterator, post_increment)
{
    constexpr RigidCoord<2, SReal> r(type::Vec<2,SReal>{0, 1}, 3_sreal);
    constexpr auto it = r.begin()++;
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidCoord2Iterator, for_loop)
{
    constexpr RigidCoord<2, SReal> coord(type::Vec<2,SReal>{0, 1}, 2_sreal);

    sofa::Size i {};
    for (const auto& r : coord)
    {
        EXPECT_FLOATINGPOINT_EQ(r, static_cast<SReal>(i++));
    }
}
}
