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
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;
#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/Link.h>
using sofa::core::objectmodel::MultiLink ;
using sofa::core::objectmodel::BaseLink ;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

using sofa::core::objectmodel::MultiLink;
class MultiLink_test: public BaseTest
{
public:
};

TEST_F(MultiLink_test, checkValidReadWithStorePath )
{
    MultiLink<BaseObject, BaseObject, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> slink ;
    /// this should not return an error as the link resolution can be done in a lazy way;
    EXPECT_TRUE(slink.read("@/node/object"));
    EXPECT_EQ(slink.getSize(), 1);
    EXPECT_EQ(slink.getPath(0), "@/node/object");
    EXPECT_EQ(slink.get(0), nullptr);

    EXPECT_TRUE(slink.read("@/node/object1 @/node/object2"));
    EXPECT_EQ(slink.getSize(), 2);
    EXPECT_EQ(slink.getPath(0), "@/node/object1");
    EXPECT_EQ(slink.getPath(1), "@/node/object2");
    EXPECT_EQ(slink.get(0), nullptr);
    EXPECT_EQ(slink.get(1), nullptr);
}

TEST_F(MultiLink_test, checkValidReadWithoutStorePath )
{
    MultiLink<BaseObject, BaseObject, BaseLink::FLAG_STRONGLINK> slink ;
    /// this should not return an error as the link resolution can be done in a lazy way;
    EXPECT_TRUE(slink.read("@/node/object"));
    EXPECT_EQ(slink.getSize(), 0);

    EXPECT_TRUE(slink.read("@/node/object1 @/node/object2"));
    EXPECT_EQ(slink.getSize(), 0);
}
