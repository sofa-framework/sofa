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

#include <sofa/core/reflection/ClassInfoBuilder.h>
using sofa::core::reflection::ClassInfo;
using sofa::core::reflection::ClassId;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

namespace classbuilder_test
{

class MyType1 : public BaseObject
{
public:
    SOFA_CLASS(MyType1, BaseObject);
};

TEST(ClassInfoBuilder_test, testUnregisteredType)
{
    /// Register the types (and their parent types)
    sofa::core::reflection::ClassInfoBuilder::GetOrBuildClassInfo<Base>("test");
    sofa::core::reflection::ClassInfoBuilder::GetOrBuildClassInfo<MyType1>("test");

    const ClassInfo* baseclass = sofa::core::reflection::Class::GetClassId<Base>().getClassInfo();
    const ClassInfo* baseoclass = sofa::core::reflection::Class::GetClassId<BaseObject>().getClassInfo();
    const ClassInfo* infoA1 = sofa::core::reflection::Class::GetClassId<MyType1>().getClassInfo();

    /// If the following three works, it means that the ClassInfoBuilder was able to recursively explore
    /// the reflection data structure to generate and register the corresponding classe info.
    ASSERT_EQ(infoA1->parents.size(), 1)
            << "MyType1 should have BasObject as parent";
    ASSERT_EQ(infoA1->parents[0],  baseoclass)
            << "MyType1 should have BasObject as parent: " << infoA1->parents[0]->className;
    ASSERT_EQ((infoA1->parents[0])->parents.size(), 1)
            << "BaseObject should have Base as parent";
    ASSERT_EQ((infoA1->parents[0])->parents[0], baseclass)
            << "BaseObject should have Base as parent" << infoA1->parents[0]->parents[0]->className;

}

}
