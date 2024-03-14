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
#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;
using sofa::core::RegisterObject;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

namespace
{

template<class Type>
class TestObject : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TestObject, Type), sofa::core::objectmodel::BaseObject);
};

template<class Type>
class TestObject2 : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TestObject2, Type), sofa::core::objectmodel::BaseObject);
};

int A = RegisterObject("Dummy test object.")
        .add< TestObject<int> >();
int B1 = RegisterObject("Dummy test object.")
        .add< TestObject<double> >();
int B2 = RegisterObject("Dummy test object.")
        .add< TestObject<long> >();
int B3 = RegisterObject("Dummy test object.")
        .add< TestObject2<int> >();

class ObjectFactory_test: public BaseTest
{
public:
    void testDuplicatedRegistration()
    {       
        EXPECT_MSG_EMIT(Warning);
        const int C = RegisterObject("Already registered object.")
                    .add< TestObject<long> >();
        SOFA_UNUSED(C);
    }

    void testValidAlias()
    {
        ASSERT_TRUE(ObjectFactory::getInstance()->addAlias("FirstAlias", "TestObject"));
    }

    void testInvalidAlias()
    {
        EXPECT_MSG_EMIT(Error);
        ASSERT_FALSE(ObjectFactory::getInstance()->addAlias("InvalidAlias", "NoWhere"));
    }

    void testDuplicatedAlias()
    {
        EXPECT_MSG_EMIT(Error);
        ASSERT_TRUE(ObjectFactory::getInstance()->addAlias("DuplicatedAlias", "TestObject"));
        ASSERT_FALSE(ObjectFactory::getInstance()->addAlias("DuplicatedAlias", "TestObject"));
    }

    void testHasCreator()
    {
        // validate the hasCreator method with use of alias when the alias is pointing
        // to two different entries in the factory
        ASSERT_TRUE(ObjectFactory::HasCreator("TestObject"));
        ASSERT_TRUE(ObjectFactory::HasCreator("TestObject2"));
   }
};

TEST_F(ObjectFactory_test, testDuplicatedRegistration)
{
    this->testDuplicatedRegistration();
}

TEST_F(ObjectFactory_test, testValidAlias )
{
    this->testValidAlias();
}

TEST_F(ObjectFactory_test, testInvalidAlias )
{
    this->testInvalidAlias();
}

TEST_F(ObjectFactory_test, testDuplicatedAlias )
{
    this->testDuplicatedAlias();
}


TEST_F(ObjectFactory_test, testHasCreator )
{
    this->testHasCreator();
}

}// namespace sofa
