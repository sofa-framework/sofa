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

#include <sofa/defaulttype/TypeInfoRegistryTools.h>
using sofa::defaulttype::TypeInfoRegistryTools;
using sofa::defaulttype::TypeInfoType;

#include "DataMockup.h"

TEST(TypeInfo, type_registration_with_implicit_declaration)
{
    DataMockup<double> dataDouble;
    ASSERT_NE(dataDouble.getTypeInfo(), nullptr);
    ASSERT_FALSE(dataDouble.getTypeInfo()->ValidInfo());
    EXPECT_EQ(dataDouble.getTypeInfo()->name(), "double");
    EXPECT_EQ(dataDouble.getTypeInfo()->getCompilationTarget(), "SofaDefaultType");

    DataMockup<int> dataInt;
    ASSERT_NE(dataInt.getTypeInfo(), nullptr);
    ASSERT_FALSE(dataInt.getTypeInfo()->ValidInfo());
    EXPECT_EQ(dataInt.getTypeInfo()->name(), "int");
    EXPECT_EQ(dataInt.getTypeInfo()->getCompilationTarget(), "SofaDefaultType");
}

