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

int A = RegisterObject("Loads a plugin and import its content into the current namespace. ")
        .add< TestObject<int> >()
        .addTargetName("FirstNameSpace");
int B1 = RegisterObject("Loads a plugin and import its content into the current namespace. ")
        .add< TestObject<double> >()
        .addTargetName("SecondNameSpace");
int B2 = RegisterObject("Loads a plugin and import its content into the current namespace. ")
        .add< TestObject<long> >()
        .addTargetName("SecondNameSpace");
int B3 = RegisterObject("Loads a plugin and import its content into the current namespace. ")
        .add< TestObject2<int> >()
        .addTargetName("SecondNameSpace");

class ObjectFactory_test: public BaseTest
{
public:
    void testDuplicatedRegistration()
    {
        {
            EXPECT_MSG_EMIT(Warning);
            int C = RegisterObject("Loads a plugin and import its content into the current namespace. ")
                    .add< TestObject<long> >()
                    .addTargetName("SecondNameSpace");
            SOFA_UNUSED(C);
        }
    }

    void testMultipleNamespace()
    {
        // There is only one object in the OneNameSpace target.
        std::vector<ObjectFactory::ClassEntry::SPtr> entries;
        ObjectFactory::getInstance()->getEntriesFromTarget(entries, "FirstNameSpace");
        ASSERT_EQ(entries.size(), 1);

        // But two the SecondNameSpace target.
        std::vector<ObjectFactory::ClassEntry::SPtr> entries2;
        ObjectFactory::getInstance()->getEntriesFromTarget(entries2, "SecondNameSpace");
        ASSERT_EQ(entries2.size(), 2);
    }

    void testObjectCreation()
    {
        ASSERT_TRUE(ObjectFactory::getInstance()->hasObjectEntry("FirstNameSpace.TestObject"));
        auto& entry1 = ObjectFactory::getInstance()->getEntry("FirstNameSpace.TestObject");
        ASSERT_EQ(entry1.className, "TestObject");
        ASSERT_EQ(entry1.compilationTarget, "FirstNameSpace");

        ASSERT_TRUE(ObjectFactory::getInstance()->hasObjectEntry("SecondNameSpace.TestObject"));
        auto& entry2 = ObjectFactory::getInstance()->getEntry("SecondNameSpace.TestObject");
        ASSERT_EQ(entry2.className, "TestObject");
        ASSERT_EQ(entry2.compilationTarget, "SecondNameSpace");
    }

    void testValidAlias()
    {
        ASSERT_TRUE(ObjectFactory::getInstance()->addAlias("FirstAlias", "FirstNameSpace.TestObject"));
    }

    void testInvalidAlias()
    {
        {
            EXPECT_MSG_EMIT(Error);
            ASSERT_FALSE(ObjectFactory::getInstance()->addAlias("InvalidAlias", "NoWhere"));
        }
    }

    void testDuplicatedAlias()
    {
        EXPECT_MSG_NOEMIT(Error);
        ASSERT_TRUE(ObjectFactory::getInstance()->addAlias("DuplicatedAlias", "FirstNameSpace.TestObject"));
        ASSERT_TRUE(ObjectFactory::getInstance()->addAlias("DuplicatedAlias", "FirstNameSpace.TestObject"));
    }

    void testHasCreator()
    {
        // validate the hasCreator method without use of alias
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("FirstNameSpace.TestObject", "int"));
        ASSERT_FALSE(ObjectFactory::getInstance()->hasCreator("FirstNameSpace.TestObject", "double"));
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("SecondNameSpace.TestObject", "double"));
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("SecondNameSpace.TestObject", "long"));
        ASSERT_FALSE(ObjectFactory::getInstance()->hasCreator("SecondNameSpace.TestObject", "int"));

        // validate the hasCreator method with use of alias when the alias is pointing
        // to two different entries in the factory
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("TestObject", "long"));
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("TestObject", "int"));
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("TestObject", "double"));
        ASSERT_FALSE(ObjectFactory::getInstance()->hasCreator("TestObject", "float"));
    }

    void testAutomaticAliasCreationForBackwardCompatibility()
    {
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("FirstNameSpace.TestObject"));
        ASSERT_TRUE(ObjectFactory::getInstance()->hasCreator("TestObject"));
    }

};

TEST_F(ObjectFactory_test, testDuplicatedRegistration)
{
    this->testDuplicatedRegistration();
}

TEST_F(ObjectFactory_test, testAliasToMultipleNamespace )
{
    this->testMultipleNamespace();
}

TEST_F(ObjectFactory_test, testObjectCreation )
{
    this->testObjectCreation();
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

TEST_F(ObjectFactory_test, testAutomaticAliasCreationForBackwardCompatibility )
{
    this->testAutomaticAliasCreationForBackwardCompatibility();
}

}// namespace sofa
