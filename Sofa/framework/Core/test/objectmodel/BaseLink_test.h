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

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

/***********************************************************************************
 * This is checking that the predicates about BaseLink are still valid in an
 * inhertited type
 ***********************************************************************************/
template<class Link>
class FakeObject : public BaseObject
{
public:
    FakeObject() : BaseObject()
    {       
    }
};

template<class Link>
class BaseLinkTests : public BaseTest
{
public:
    Link link1;
    Link link2;
    FakeObject<Link> object1;
    FakeObject<Link> object2;
};

TYPED_TEST_SUITE_P(BaseLinkTests);

TYPED_TEST_P(BaseLinkTests, checkOwnerSetGet)
{
    this->link1.setOwner(&this->object1);
    EXPECT_EQ(this->link1.getOwnerBase(), &this->object1);
}

REGISTER_TYPED_TEST_SUITE_P(BaseLinkTests,
                            checkOwnerSetGet);

