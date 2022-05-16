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
#include <sofa/testing/BaseTest.h>
#include <sofa/helper/TagFactory.h>

using sofa::helper::TagFactory;

TEST(TagFactory_test, initialization)
{
    EXPECT_EQ(TagFactory::getName(0), "0");
    EXPECT_EQ(TagFactory::getName(1), "Visual");

    // id 14 is not yet in the tag list: an empty string is returned
    EXPECT_EQ(TagFactory::getName(14), "");
}

TEST(TagFactory_test, addTag)
{
    // Get id of new Tags. Will be added to the list
    EXPECT_EQ(TagFactory::getID("1"), 2);
    EXPECT_EQ(TagFactory::getID("2"), 3);
    EXPECT_EQ(TagFactory::getID("4"), 4);
    EXPECT_EQ(TagFactory::getID("2"), 3);

    EXPECT_EQ(TagFactory::getID("foo"), 5);
    EXPECT_EQ(TagFactory::getID("bar"), 6);
    EXPECT_EQ(TagFactory::getID("foo"), 5);

    // Get id of existing Tags
    EXPECT_EQ(TagFactory::getID("0"), 0);
    EXPECT_EQ(TagFactory::getID("Visual"), 1);
}
