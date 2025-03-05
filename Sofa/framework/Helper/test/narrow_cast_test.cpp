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
#include <sofa/helper/narrow_cast.h>
#include <gtest/gtest.h>

namespace sofa
{
    TEST(narrow_cast_test, in_range)
    {
        {
            size_t a = 0;
            const int b = 0;
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
        }

        {
            const size_t a = 0;
            const int b = 0;
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
        }

        {
            const size_t a = 0;
            const int b = 0;
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
        }

        {
            constexpr size_t a = std::numeric_limits<int>::max();
            constexpr int b = std::numeric_limits<int>::max();
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
        }

    }

    // Situations where narrow conversion changes the input value
    TEST(narrow_cast_test, out_of_range)
    {
        //size_t -> int: out of range
        {
            constexpr size_t a = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
#if !defined(NDEBUG)
            EXPECT_THROW(sofa::helper::narrow_cast<int>(a), sofa::helper::narrowing_error);
#else
            constexpr int b = std::numeric_limits<int>::min(); // overflow
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
#endif
        }

        //unsigned int -> int : out of range
        {
            constexpr unsigned int a = static_cast<unsigned int>(std::numeric_limits<int>::max()) + 1;
#if !defined(NDEBUG)
            EXPECT_THROW(sofa::helper::narrow_cast<int>(a), sofa::helper::narrowing_error);
#else
            constexpr int b = std::numeric_limits<int>::min(); // overflow
            EXPECT_EQ(sofa::helper::narrow_cast<int>(a), b);
#endif
        }

        //int -> unsigned int : negative value
        {
            constexpr int a = -1;
#if !defined(NDEBUG)
            EXPECT_THROW(sofa::helper::narrow_cast<unsigned int>(a), sofa::helper::narrowing_error);
#else
            constexpr int b = std::numeric_limits<unsigned int>::max();
            EXPECT_EQ(sofa::helper::narrow_cast<unsigned int>(a), b);
#endif
        }

        //int -> short: out of range
        {
            constexpr int a = static_cast<size_t>(std::numeric_limits<short>::max()) + 1;
#if !defined(NDEBUG)
            EXPECT_THROW(sofa::helper::narrow_cast<short>(a), sofa::helper::narrowing_error);
#else
            constexpr int b = std::numeric_limits<short>::min(); // overflow
            EXPECT_EQ(sofa::helper::narrow_cast<short>(a), b);
#endif
        }
    }
}
