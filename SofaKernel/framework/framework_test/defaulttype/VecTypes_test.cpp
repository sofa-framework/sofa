/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/defaulttype/VecTypes.h>
#include <gtest/gtest.h>

using sofa::defaulttype::ResizableExtVector;
using sofa::defaulttype::DefaultAllocator;

class ResizableExtVectorTest : public ::testing::Test
{
protected:
    ResizableExtVectorTest()
    {
        v.resize(10);
        int i = 0;
        for(ResizableExtVector<int>::iterator elem = v.begin(), end = v.end(); elem != end; ++elem, ++i)
        {
            *elem = i;
        }
    }
    virtual ~ResizableExtVectorTest()
    {
    }
    ResizableExtVector<int> v;
};


TEST(VecTypesTest, testDefaultConstructor)
{
    ResizableExtVector<int> v;
    EXPECT_EQ(v.empty(), true);
    EXPECT_EQ(v.size(), 0u);
    EXPECT_EQ(v.getData(), (int*)0);
    EXPECT_EQ(v.begin(), v.end());
}

TEST_F(ResizableExtVectorTest, testSizing)
{
    EXPECT_EQ(v.empty(), false);
    EXPECT_EQ(v.size(), 10u);
}

TEST_F(ResizableExtVectorTest, testClear)
{
    v.clear();
    EXPECT_EQ(v.empty(), true);
    EXPECT_EQ(v.size(), 0u);
}

TEST_F(ResizableExtVectorTest, testIterators)
{
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(v[i], i);
    }
}

TEST_F(ResizableExtVectorTest, testIncreaseSize)
{
    v.resize(20);
    EXPECT_EQ(v.size(), 20u);
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(v[i], i);
    }
}

TEST_F(ResizableExtVectorTest, testReduceSize)
{
    v.resize(5);
    EXPECT_EQ(v.size(), 5u);
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(v[i], i);
    }
}

TEST_F(ResizableExtVectorTest, testSetNullAllocator)
{
    v.setAllocator(0);
    EXPECT_EQ(v.empty(), true);
    EXPECT_EQ(v.size(), 0u);
    EXPECT_EQ(v.getData(), (int*)0);
}

TEST_F(ResizableExtVectorTest, testSetOtherAllocator)
{
    v.setAllocator(new DefaultAllocator<int>);
    EXPECT_EQ(v.empty(), false);
    EXPECT_EQ(v.size(), 10u);
    EXPECT_TRUE(v.getData() != 0);

    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(v[i], i);
    }
}

TEST_F(ResizableExtVectorTest, testCopyConstructor_Sizing)
{
    ResizableExtVector<int> v2 = v;
    EXPECT_EQ(v2.empty(), false);
    EXPECT_EQ(v2.size(), 10u);
}

TEST_F(ResizableExtVectorTest, testCopyConstructor_Separation)
{
    ResizableExtVector<int> v2 = v;
    v.resize(0);
    EXPECT_EQ(v2.empty(), false);
    EXPECT_EQ(v2.size(), 10u);
}


TEST_F(ResizableExtVectorTest, testCopyConstructor_Data)
{
    ResizableExtVector<int> v2 = v;
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(v2[i], i);
    }
}
