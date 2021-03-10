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

#include <sofa/core/reflection/Class.h>
using sofa::core::reflection::Class;

#include <sofa/core/reflection/ClassInfoRepository.h>
using sofa::core::reflection::ClassInfoRepository;
using sofa::core::reflection::ClassInfo;
using sofa::core::reflection::ClassId;

#include <sofa/core/reflection/ClassInfoBuilder.h>
using sofa::core::reflection::ClassInfoBuilder;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

namespace classinforepository_test
{
    class MyType1 : public BaseObject
    {
    public:
        SOFA_CLASS(MyType1, BaseObject);
    };

    TEST(ClassInfoRepository_test, testRegistration)
    {
        size_t sizeAtInit = ClassInfoRepository::GetRegisteredTypes().size();

        /// Allocates a new id,
        ClassId cid = Class::GetClassId<MyType1>();
        ASSERT_GT(cid.id, 0);

        /// Check that there is now an empty but non null class info returned for the corresponding cid
        ASSERT_ANY_THROW(ClassInfoRepository::Get(cid)) << "Every cid should correspond to a non null classinfo before initialization.";

        /// Create a new Class Info and register it.
        auto cinfo = ClassInfoBuilder::GetOrBuildClassInfo<MyType1>("test");

        /// Register in the repository at given index.
        int retvalue = ClassInfoRepository::Set(cid, cinfo);
        ASSERT_EQ(retvalue, -1) << "Value: -1, there was already a valid registered typeinfo. Value: 1, no problem. Value: 2,"
                                  " new cinfo overrides an existing one. Here it should returns -1 as the cinfo was added by the ClassInfoBuilder";

        size_t sizeAtEnd = ClassInfoRepository::GetRegisteredTypes().size();
        ASSERT_NE(sizeAtInit, sizeAtEnd) << "The repository seems to have not changed in size. The registration probably failed";
    }
}
