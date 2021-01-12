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

#include <sofa/core/objectmodel/BaseLink.h>
using sofa::core::objectmodel::BaseLink ;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

/***********************************************************************************
 * This is checking that the predicates about BaseLink are still valid in an
 * inhertited type
 ***********************************************************************************/
class FakeObject : public BaseObject
{
public:
    FakeObject() : BaseObject() {}
};

template<class Link>
class BaseLinkTests : public BaseTest
{
public:
    Link link1;
    Link link2;
    FakeObject object1;
    FakeObject object2;
};

TYPED_TEST_SUITE_P(BaseLinkTests);

TYPED_TEST_P(BaseLinkTests, checkOwnerShipTransfer)
{
    this->link1.setOwner(&this->object1);
    EXPECT_EQ(this->link1.getOwnerBase(), &this->object1);
    ASSERT_EQ(this->object1.getLinks().size(), 1);
    EXPECT_EQ(this->object1.getLinks()[0], &this->link1);

    this->link2.setOwner(&this->object2);
    EXPECT_EQ(this->link2.getOwnerBase(), &this->object2);
    EXPECT_EQ(this->object1.getLinks().size(), 0);
    EXPECT_EQ(this->object2.getLinks().size(), 0);
    EXPECT_EQ(this->object2.getLinks()[0], &this->link2);
}

TYPED_TEST_P(BaseLinkTests, checkRead)
{
    EXPECT_TRUE(this->link1.read("@/node/object1"));
    EXPECT_EQ(this->link1.getSize(), 1);
    EXPECT_NE(this->link1.getLinkedBase(), nullptr);

    if(this->link1.storePath())
    {
        EXPECT_EQ(this->link1.getLinkedPath(), "/node/object1");
    }
}

TYPED_TEST_P(BaseLinkTests, checkReadWithMultipleLinkPath)
{
    if(this->link1.isMultiLink())
    {
        EXPECT_TRUE(this->link1.read("@/node/object1 @/node/object2"));
        EXPECT_EQ(this->link1.getSize(), 2);
        EXPECT_NE(this->link1.getLinkedBase(0), nullptr);
        EXPECT_NE(this->link1.getLinkedBase(1), nullptr);
    }else
    {
        EXPECT_FALSE(this->link1.read("@/node/object1 @/node/object2"));
        EXPECT_EQ(this->link1.getSize(), 0);
    }

    if(this->link1.storePath())
    {
        EXPECT_EQ(this->link1.getLinkedPath(), "/node/object1");
    }
}


REGISTER_TYPED_TEST_SUITE_P(BaseLinkTests, checkOwnerShipTransfer, checkRead, checkReadWithMultipleLinkPath);

