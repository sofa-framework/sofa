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
#include <sofa/helper/IotaView.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(IotaView, loop)
{
    const auto range = sofa::helper::IotaView{0, 10};

    int i = 0;
    for (const auto value : range)
    {
        EXPECT_EQ(value, i);
        ++i;
    }
}

TEST(IotaView, empty)
{
    {
        const auto range = sofa::helper::IotaView{0, 10};
        EXPECT_FALSE(range.empty());
    }
    {
        const auto range = sofa::helper::IotaView{0, 0};
        EXPECT_TRUE(range.empty());
    }
}

TEST(IotaView, size)
{
    const auto range = sofa::helper::IotaView{0, 10};
    EXPECT_EQ(range.size(), 10);
}

TEST(IotaView, access)
{
    const auto range = sofa::helper::IotaView{4, 10};
    EXPECT_EQ(range[0], 4);
    EXPECT_EQ(range[9], 4+9);
}

}
