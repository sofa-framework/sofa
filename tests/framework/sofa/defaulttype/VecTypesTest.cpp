/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Tests                                 *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/defaulttype/VecTypes.h>
#include <boost/test/auto_unit_test.hpp>

using sofa::defaulttype::ResizableExtVector;
using sofa::defaulttype::DefaultAllocator;

struct ResizableExtVectorFixture
{
    ResizableExtVectorFixture()
    {
        v.resize(10);
        int i = 0;
        for(ResizableExtVector<int>::iterator elem = v.begin(), end = v.end(); elem != end; ++elem, ++i)
        {
            *elem = i;
        }
    }
    ~ResizableExtVectorFixture()
    {
    }
    ResizableExtVector<int> v;
};


BOOST_AUTO_TEST_CASE(testDefaultConstructor)
{
    ResizableExtVector<int> v;
    BOOST_CHECK_EQUAL(v.empty(), true);
    BOOST_CHECK_EQUAL(v.size(), 0);
    BOOST_CHECK_EQUAL(v.getData(), (int*)0);
    BOOST_CHECK_EQUAL(v.begin(), v.end());
}

BOOST_FIXTURE_TEST_CASE(testSizing, ResizableExtVectorFixture)
{
    BOOST_CHECK_EQUAL(v.empty(), false);
    BOOST_CHECK_EQUAL(v.size(), 10);
}

BOOST_FIXTURE_TEST_CASE(testClear, ResizableExtVectorFixture)
{
    v.clear();
    BOOST_CHECK_EQUAL(v.empty(), true);
    BOOST_CHECK_EQUAL(v.size(), 0);
}

BOOST_FIXTURE_TEST_CASE(testIterators, ResizableExtVectorFixture)
{
    for(int i = 0; i < 10; ++i)
    {
        BOOST_CHECK_EQUAL(v[i], i);
    }
}

BOOST_FIXTURE_TEST_CASE(testIncreaseSize, ResizableExtVectorFixture)
{
    v.resize(20);
    BOOST_CHECK_EQUAL(v.size(), 20);
    for(int i = 0; i < 10; ++i)
    {
        BOOST_CHECK_EQUAL(v[i], i);
    }
}

BOOST_FIXTURE_TEST_CASE(testReduceSize, ResizableExtVectorFixture)
{
    v.resize(5);
    BOOST_CHECK_EQUAL(v.size(), 5);
    for(int i = 0; i < 5; ++i)
    {
        BOOST_CHECK_EQUAL(v[i], i);
    }
}

BOOST_FIXTURE_TEST_CASE(testSetNullAllocator, ResizableExtVectorFixture)
{
    v.setAllocator(0);
    BOOST_CHECK_EQUAL(v.empty(), true);
    BOOST_CHECK_EQUAL(v.size(), 0);
    BOOST_CHECK_EQUAL(v.getData(), (int*)0);
}

BOOST_FIXTURE_TEST_CASE(testSetOtherAllocator, ResizableExtVectorFixture)
{
    v.setAllocator(new DefaultAllocator<int>);
    BOOST_CHECK_EQUAL(v.empty(), false);
    BOOST_CHECK_EQUAL(v.size(), 10);
    BOOST_CHECK(v.getData() != 0);

    for(int i = 0; i < 10; ++i)
    {
        BOOST_CHECK_EQUAL(v[i], i);
    }
}

BOOST_FIXTURE_TEST_CASE(testCopyConstructor_Sizing, ResizableExtVectorFixture)
{
    ResizableExtVector<int> v2 = v;
    BOOST_CHECK_EQUAL(v2.empty(), false);
    BOOST_CHECK_EQUAL(v2.size(), 10);
}

BOOST_FIXTURE_TEST_CASE(testCopyConstructor_Separation, ResizableExtVectorFixture)
{
    ResizableExtVector<int> v2 = v;
    v.resize(0);
    BOOST_CHECK_EQUAL(v2.empty(), false);
    BOOST_CHECK_EQUAL(v2.size(), 10);
}


BOOST_FIXTURE_TEST_CASE(testCopyConstructor_Data, ResizableExtVectorFixture)
{
    ResizableExtVector<int> v2 = v;
    for(int i = 0; i < 10; ++i)
    {
        BOOST_CHECK_EQUAL(v2[i], i);
    }
}
