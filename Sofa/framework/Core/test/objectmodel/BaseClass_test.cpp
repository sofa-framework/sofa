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
using sofa::core::objectmodel::Base ;

#include <sofa/helper/NameDecoder.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

namespace sofa{
namespace another_namespace{

class EmptyObject : public BaseObject
{
public:
    SOFA_CLASS(EmptyObject, BaseObject) ;
};

}
}

namespace sofa{
namespace numbered_namespace_123{

class NumberedClass123 : public BaseObject
{
public:
    SOFA_CLASS(NumberedClass123, BaseObject) ;
};

class NumberedClass456 : public another_namespace::EmptyObject
{
public:
    SOFA_CLASS(NumberedClass456, another_namespace::EmptyObject) ;
};

class CustomName123 : public BaseObject
{
public:
    SOFA_CLASS(CustomName123, BaseObject) ;

    static const std::string GetCustomClassName(){ return "ClassWithACustomName"; }
    static const std::string GetCustomTemplateName(){ return "ClassWithACustomTemplate"; }

    template<class T>
    static const std::string className(){ return "TEST TEST"; }
};

class CustomNameOldWay : public BaseObject
{
public:
    SOFA_CLASS(CustomNameOldWay, BaseObject) ;

    static const std::string className(const CustomNameOldWay* =nullptr){ return "ClassWithACustomNameOldWay"; }

    template<class T>
    static const std::string shortName(const T*)
    {
        return "MECHANICAL";
    }

};

}
}

class DataOne { public: static std::string Name(){ return "One" ;} };
class DataTwo { public: static std::string Name(){ return "Two" ;} };
class NotAType {};

template<class DataType1>
class DefaultTemplate1 : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DefaultTemplate1, DataType1), BaseObject) ;
};

template<class DataType1, class DataType2>
class DefaultTemplate2 : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DefaultTemplate2, DataType1, DataType2), BaseObject) ;
};

template<class DataType1, class DataType2, class NotAType>
class DefaultTemplate3 : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(DefaultTemplate3, DataType1, DataType2, NotAType), BaseObject) ;
};

template<class DataType1, class DataType2, class NotAType>
class NotDefaultTemplate : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(NotDefaultTemplate, DataType1, DataType2, NotAType), BaseObject) ;

    static const std::string GetCustomTemplateName(){ return "non,oui"; }
};

template<class TDataType1>
class OuterClass : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OuterClass, TDataType1), BaseObject);

    template<class TDataType2>
    class InnerClass : public BaseObject
    {
    public:
        SOFA_CLASS(SOFA_TEMPLATE(InnerClass, TDataType2), BaseObject);
    };
};

class BaseClass_test: public BaseTest
{
public:

    sofa::another_namespace::EmptyObject m_ptr1;
    sofa::numbered_namespace_123::NumberedClass123 m_ptr2;
    sofa::numbered_namespace_123::NumberedClass456 m_ptr3;
    sofa::numbered_namespace_123::CustomName123 m_ptr4;
    sofa::numbered_namespace_123::CustomNameOldWay m_ptr5;
    DefaultTemplate1<DataOne> m_ptr7;
    DefaultTemplate2<DataOne, DataTwo> m_ptr8;
    DefaultTemplate3<DataOne, DataTwo, NotAType> m_ptr9;
    OuterClass<DataOne> m_ptr10;
    OuterClass<DataOne>::InnerClass<DataTwo> m_ptr11;
    DefaultTemplate1<float> m_ptr12;
    DefaultTemplate1<sofa::type::vector<float>> m_ptr13;
    DefaultTemplate1<sofa::type::Vec3d> m_ptr14;
    DefaultTemplate1<sofa::defaulttype::Vec3dTypes> m_ptr15;

    sofa::core::objectmodel::Base* m_baseptr1 {&m_ptr1};
    sofa::core::objectmodel::Base* m_baseptr2 {&m_ptr2};
    sofa::core::objectmodel::Base* m_baseptr3 {&m_ptr3};
    sofa::core::objectmodel::Base* m_baseptr4 {&m_ptr4};
    sofa::core::objectmodel::Base* m_baseptr5 {&m_ptr5};
};

///
/// tests that all the BaseClass returned from GetClass function are refering to the same
/// BaseClass instance.
///
TEST_F(BaseClass_test, checkClassEquivalence  )
{
    EXPECT_EQ(sofa::another_namespace::EmptyObject::GetClass(), m_ptr1.getClass());
    EXPECT_EQ(sofa::another_namespace::EmptyObject::GetClass(), m_baseptr1->getClass());

    EXPECT_EQ(sofa::numbered_namespace_123::NumberedClass123::GetClass(), m_ptr2.getClass());
    EXPECT_EQ(sofa::numbered_namespace_123::NumberedClass123::GetClass(), m_baseptr2->getClass());

    EXPECT_EQ(sofa::numbered_namespace_123::NumberedClass456::GetClass(), m_ptr3.getClass());
    EXPECT_EQ(sofa::numbered_namespace_123::NumberedClass456::GetClass(), m_baseptr3->getClass());
}

TEST_F(BaseClass_test, checkStaticClassName  )
{
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<decltype(m_ptr1)>(),"EmptyObject");
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<decltype(m_ptr2)>(),"NumberedClass123");
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<decltype(m_ptr3)>(),"NumberedClass456");

    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<sofa::another_namespace::EmptyObject>(),"EmptyObject");
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<sofa::numbered_namespace_123::NumberedClass123>(),"NumberedClass123");
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<sofa::numbered_namespace_123::NumberedClass456>(),"NumberedClass456");
}

TEST_F(BaseClass_test, checkDynamicClassName  )
{
    EXPECT_EQ(m_ptr1.getClassName(),"EmptyObject") ;
    EXPECT_EQ(m_ptr2.getClassName(),"NumberedClass123") ;
    EXPECT_EQ(m_ptr3.getClassName(),"NumberedClass456") ;
}

TEST_F(BaseClass_test, checkDynamicCustomName  )
{
    EXPECT_EQ(m_ptr4.getClassName(),"ClassWithACustomName") ;
}

TEST_F(BaseClass_test, checkDynamicGetCustomTemplateName  )
{
    EXPECT_EQ(m_ptr4.getTemplateName(),"ClassWithACustomTemplate") ;
}

TEST_F(BaseClass_test, checkDynamicClassNameOnBase  )
{
    ASSERT_EQ(m_baseptr1->getClassName(),"EmptyObject") ;
    ASSERT_EQ(m_baseptr2->getClassName(),"NumberedClass123") ;
    ASSERT_EQ(m_baseptr3->getClassName(),"NumberedClass456") ;
}

TEST_F(BaseClass_test, checkDynamicClassCustomNameOnBase  )
{
    ASSERT_EQ(m_baseptr4->getClassName(),"ClassWithACustomName") ;
}

TEST_F(BaseClass_test, checkDynamicGetCustomTemplateNameOnBase  )
{
    ASSERT_EQ(m_baseptr4->getTemplateName(),"ClassWithACustomTemplate") ;
}

TEST_F(BaseClass_test, checkStaticDefaultTemplate  )
{
    EXPECT_EQ(m_ptr7.getClassName(),"DefaultTemplate1") ;
    EXPECT_EQ(m_ptr7.getTemplateName(),"One") ;

    EXPECT_EQ(m_ptr8.getClassName(),"DefaultTemplate2") ;
    EXPECT_EQ(m_ptr8.getTemplateName(),"One,Two") ;

    EXPECT_EQ(m_ptr9.getClassName(),"DefaultTemplate3") ;
    EXPECT_EQ(m_ptr9.getTemplateName(),"One,Two,NotAType") ;

    EXPECT_EQ(m_ptr12.getTemplateName(),"f") ;
    EXPECT_EQ(m_ptr13.getTemplateName(),"vector<f>") ;
    EXPECT_EQ(m_ptr14.getTemplateName(),"Vec3d") ;
    EXPECT_EQ(m_ptr15.getTemplateName(),"Vec3d") ;
}

TEST_F(BaseClass_test, checkStaticDefaultTemplateOverridenByCustom  )
{
    const NotDefaultTemplate<DataOne, DataTwo, NotAType> ptr;
    EXPECT_EQ(ptr.getClassName(),"NotDefaultTemplate") ;
    EXPECT_EQ(ptr.getTemplateName(),"non,oui") ;
}

TEST_F(BaseClass_test, checkNameSpace)
{
    ASSERT_EQ(m_ptr1.getNameSpaceName(),"sofa::another_namespace") ;
    ASSERT_EQ(m_ptr2.getNameSpaceName(),"sofa::numbered_namespace_123") ;
    ASSERT_EQ(m_ptr3.getNameSpaceName(),"sofa::numbered_namespace_123") ;
}

////
TEST_F(BaseClass_test, checkStaticGetCustomClassNameOldWay  )
{
    EXPECT_EQ(m_ptr5.getClass()->shortName,"MECHANICAL") ;
    EXPECT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getShortName<sofa::numbered_namespace_123::CustomNameOldWay>(), "MECHANICAL" );
    ASSERT_EQ(m_ptr5.getClassName(),"ClassWithACustomNameOldWay") ;
    ASSERT_EQ(m_baseptr5->getClassName(),"ClassWithACustomNameOldWay") ;
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<decltype(m_ptr5)>(),"ClassWithACustomNameOldWay") ;
    ASSERT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getClassName<sofa::numbered_namespace_123::CustomNameOldWay>(),"ClassWithACustomNameOldWay") ;
}

TEST_F(BaseClass_test, checkNestedClass)
{
    EXPECT_EQ(m_ptr10.getClassName(),"OuterClass") ;
    EXPECT_EQ(m_ptr10.getTemplateName(),"One") ;

    EXPECT_EQ(m_ptr11.getClassName(),"InnerClass") ;
    EXPECT_EQ(m_ptr11.getTemplateName(),"Two") ;
}
