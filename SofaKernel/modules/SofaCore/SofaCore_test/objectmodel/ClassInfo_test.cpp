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
using sofa::core::objectmodel::Base;

#include <sofa/core/reflection/ClassId.h>
using sofa::core::reflection::ClassId;

#include <sofa/core/reflection/ClassInfo.h>
using sofa::core::reflection::ClassInfo;

#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

namespace test
{

class UnregisteredType : public BaseObject {};
class RegisteredType : public BaseObject {};

static int registerT = RegisterObject("test object")
        .add<RegisteredType>();

/// A type the has not been registered in the ClassInfoRepository always returns a MissingClassInfo
/// type.
TEST(ClassInfo_test, checkClassInfoForUnregistreredType)
{
    const ClassId& classid = sofa::core::reflection::Class::GetClassId<UnregisteredType>();
    ASSERT_ANY_THROW(classid.getClassInfo());
}

/// A type the has been registered in the ClassInfoRepository should returns a fully
/// defined class info.
TEST(ClassInfo_test, checkClassInfoForRegistreredType)
{
    const ClassId& classid = sofa::core::reflection::Class::GetClassId<RegisteredType>();
    ASSERT_EQ(classid.getClassInfo()->className, "RegisteredType");
}

TEST(ClassInfo_test, checkDynamicCastToBase)
{
    RegisteredType t1;
    UnregisteredType t2;

    const ClassId& classid = sofa::core::reflection::Class::GetClassId<RegisteredType>();
    ASSERT_EQ(classid.getClassInfo()->dynamicCastToBase(&t1), &t1);
    ASSERT_EQ(classid.getClassInfo()->dynamicCastToBase(&t2), nullptr);
}

TEST(ClassInfo_test, checkDynamicCast)
{
    RegisteredType t1;
    UnregisteredType t2;

    const ClassId& classid = sofa::core::reflection::Class::GetClassId<BaseObject>();
    ASSERT_NE(classid.getClassInfo()->dynamicCastToBase(&t1), nullptr);
    ASSERT_NE(classid.getClassInfo()->dynamicCastToBase(&t2), nullptr);
}

TEST(ClassInfo_test, checkIsInstance)
{
    RegisteredType t1;
    UnregisteredType t2;

    const ClassId& baseclassid = sofa::core::reflection::Class::GetClassId<Base>();
    ASSERT_TRUE(baseclassid.getClassInfo()->isInstance(&t1));
    ASSERT_TRUE(baseclassid.getClassInfo()->isInstance(&t2));

    const ClassId& baseobjectclassid = sofa::core::reflection::Class::GetClassId<BaseObject>();
    ASSERT_TRUE(baseobjectclassid.getClassInfo()->isInstance(&t1));
    ASSERT_TRUE(baseobjectclassid.getClassInfo()->isInstance(&t2));

    const ClassId& classid = sofa::core::reflection::Class::GetClassId<RegisteredType>();
    ASSERT_TRUE(classid.getClassInfo()->isInstance(&t1));
    ASSERT_FALSE(classid.getClassInfo()->isInstance(&t2));
}

TEST(ClassInfo_test, checkHashParents)
{
    const ClassInfo* base_classinfo = sofa::core::reflection::Class::GetClassId<Base>().getClassInfo();
    const ClassInfo* baseobject_classinfo = sofa::core::reflection::Class::GetClassId<Base>().getClassInfo();
    const ClassInfo* reg_classinfo = sofa::core::reflection::Class::GetClassId<RegisteredType>().getClassInfo();

    /// Base introspection
    ASSERT_FALSE(base_classinfo->hasParent(reg_classinfo));

    /// BaseObject introspection
    ASSERT_FALSE(baseobject_classinfo->hasParent(reg_classinfo));
    ASSERT_TRUE(baseobject_classinfo->hasParent(base_classinfo));

    /// RegisteredType introspection
    ASSERT_TRUE(reg_classinfo->hasParent(base_classinfo));
    ASSERT_TRUE(reg_classinfo->hasParent(baseobject_classinfo));
}

TEST(ClassInfo_test, checkHasParent)
{
    const ClassId& baseclassid = sofa::core::reflection::Class::GetClassId<Base>();
    ASSERT_EQ(baseclassid.getClassInfo()->parents.size(), 0);

    const ClassId& classid = sofa::core::reflection::Class::GetClassId<RegisteredType>();
    ASSERT_EQ(classid.getClassInfo()->parents.size(), 1);
}


}
