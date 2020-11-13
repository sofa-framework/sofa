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

#include <gtest/gtest.h>

#include <sofa/defaulttype/AbstractTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo;

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
using sofa::defaulttype::IncompleteTypeInfo;

#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
using sofa::defaulttype::DataTypeInfo;
using sofa::defaulttype::DataTypeInfoDynamicWrapper;

#include <sofa/defaulttype/TypeInfoID.h>
using sofa::defaulttype::TypeInfoId;

#include <sofa/defaulttype/TypeInfoRegistry.h>
using sofa::defaulttype::TypeInfoRegistry;
using sofa::defaulttype::TypeInfoType;

#include <sofa/defaulttype/typeinfo/NoTypeInfo.h>
using sofa::defaulttype::NoTypeInfo;

#include <sofa/defaulttype/TypeInfoRegistryTools.h>
using sofa::defaulttype::TypeInfoRegistryTools;

/// Forward declaration of an object in translation unit1.
class ObjectInTranslationUnit1 {};

class ObjectInTranslationUnit2 {};
template<> class sofa::defaulttype::DataTypeInfo<ObjectInTranslationUnit2> : public IncompleteTypeInfo<ObjectInTranslationUnit2>
{
public:
    static std::string name(){ return "ObjectInTranslationUnit2"; }
    static std::string GetTypeName(){ return "ObjectInTranslationUnit2"; }
};

TEST(TypeInfoRegistryTu2, internal_set_internal_get)
{
    TypeInfoRegistry::Set(TypeInfoId::getTypeId<ObjectInTranslationUnit2>(),
                          DataTypeInfoDynamicWrapper<DataTypeInfo<ObjectInTranslationUnit2>>::get(),
                          "TranslationUnit2");

    const AbstractTypeInfo* nfo = TypeInfoRegistry::Get(TypeInfoId::getTypeId<ObjectInTranslationUnit2>());
    EXPECT_NE(nfo, nullptr);
    EXPECT_FALSE(nfo->ValidInfo());
    EXPECT_EQ(nfo->name(), std::string("ObjectInTranslationUnit2"));
    EXPECT_EQ(nfo->getCompilationTarget(), std::string("TranslationUnit2"));
}

TEST(TypeInfoRegistry, external_set_internal_get)
{
    const AbstractTypeInfo* nfo = TypeInfoRegistry::Get(TypeInfoId::getTypeId<ObjectInTranslationUnit1>());
    EXPECT_NE(nfo, nullptr);
    EXPECT_FALSE(nfo->ValidInfo());
    EXPECT_NE(nfo, NoTypeInfo::get());
    EXPECT_EQ(nfo->name(), std::string("ObjectInTranslationUnit1"));
    EXPECT_EQ(nfo->getCompilationTarget(), std::string("TranslationUnit1"));
    TypeInfoRegistryTools::dumpRegistryContentToStream(std::cout, TypeInfoType::ALL);
}
