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
#include <sofa/defaulttype/RigidDeriv.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{
using defaulttype::RigidDeriv;

TEST(RigidDeriv3Iterator, begin)
{
    static constexpr RigidDeriv<3, SReal> r;
    static constexpr auto it = r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidDeriv3Iterator, begin_begin)
{
    static constexpr RigidDeriv<3, SReal> r;
    static constexpr auto it1 = r.begin();
    static constexpr auto it2 = r.begin();
    EXPECT_EQ(it1, it2);
}

TEST(RigidDeriv3Iterator, begin_end)
{
    static constexpr RigidDeriv<3, SReal> r;
    static constexpr auto begin = r.begin();
    static constexpr auto end = r.end();
    EXPECT_NE(begin, end);
}

TEST(RigidDeriv3Iterator, pre_increment)
{
    static constexpr RigidDeriv<3, SReal> r(type::Vec<3,SReal>{0, 1, 2}, type::Vec<3, SReal>{3, 4, 5});
    static constexpr auto it = ++r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 1_sreal);
}

TEST(RigidDeriv3Iterator, post_increment)
{
    static constexpr RigidDeriv<3, SReal> r(type::Vec<3,SReal>{0, 1, 2}, type::Vec<3, SReal>{3, 4, 5});
    static constexpr auto it = r.begin()++;
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidDeriv3Iterator, for_loop)
{
    static constexpr RigidDeriv<3, SReal> coord(type::Vec<3,SReal>{0, 1, 2}, type::Vec<3, SReal>{3, 4, 5});

    sofa::Size i {};
    for (const auto& r : coord)
    {
        EXPECT_FLOATINGPOINT_EQ(r, static_cast<SReal>(i++));
    }
}


TEST(RigidDeriv2Iterator, begin)
{
    static constexpr RigidDeriv<2, SReal> r;
    static constexpr auto it = r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidDeriv2Iterator, begin_begin)
{
    static constexpr RigidDeriv<2, SReal> r;
    static constexpr auto it1 = r.begin();
    static constexpr auto it2 = r.begin();
    EXPECT_EQ(it1, it2);
}

TEST(RigidDeriv2Iterator, begin_end)
{
    static constexpr RigidDeriv<2, SReal> r;
    static constexpr auto begin = r.begin();
    static constexpr auto end = r.end();
    EXPECT_NE(begin, end);
}

TEST(RigidDeriv2Iterator, pre_increment)
{
    static constexpr RigidDeriv<2, SReal> r(type::Vec<2,SReal>{0, 1}, 2_sreal);
    static constexpr auto it = ++r.begin();
    EXPECT_FLOATINGPOINT_EQ(*it, 1_sreal);
}

TEST(RigidDeriv2Iterator, post_increment)
{
    static constexpr RigidDeriv<2, SReal> r(type::Vec<2,SReal>{0, 1}, 2_sreal);
    static constexpr auto it = r.begin()++;
    EXPECT_FLOATINGPOINT_EQ(*it, 0_sreal);
}

TEST(RigidDeriv2Iterator, for_loop)
{
    static constexpr RigidDeriv<2, SReal> coord(type::Vec<2,SReal>{0, 1}, 2_sreal);

    sofa::Size i {};
    for (const auto& r : coord)
    {
        EXPECT_FLOATINGPOINT_EQ(r, static_cast<SReal>(i++));
    }
}

}
