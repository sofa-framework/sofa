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
#include <sofa/testing/config.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/vectorData.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/defaulttype/AbstractTypeInfo.h>

using sofa::testing::BaseTest ;


namespace sofa {

using namespace core::objectmodel;

class Data_test : public BaseTest
{
public:
    Data<int> dataInt;
    Data<float> dataFloat;
    Data<bool> dataBool;
    Data<sofa::type::Vec3> dataVec3;
    Data<sofa::type::vector<sofa::type::Vec3>> dataVectorVec3;
    Data<sofa::type::vector<sofa::type::RGBAColor>> dataVectorColor;
};

TEST_F(Data_test, getValueTypeString)
{
    EXPECT_EQ(dataInt.getValueTypeString(), "i");
    EXPECT_EQ(dataFloat.getValueTypeString(), "f");
    EXPECT_EQ(dataBool.getValueTypeString(), "bool");
    EXPECT_EQ(dataVec3.getValueTypeString(), "Vec3d");
    EXPECT_EQ(dataVectorVec3.getValueTypeString(), "vector<Vec3d>");
    EXPECT_EQ(dataVectorColor.getValueTypeString(), "vector<RGBAColor>");
}

TEST_F(Data_test, getNameWithValueTypeInfo)
{
    EXPECT_EQ(dataInt.getValueTypeInfo()->name(), "i");
    EXPECT_EQ(dataFloat.getValueTypeInfo()->name(), "f");
    EXPECT_EQ(dataBool.getValueTypeInfo()->name(), "bool");
    EXPECT_EQ(dataVec3.getValueTypeInfo()->name(), "Vec3d");
    EXPECT_EQ(dataVectorVec3.getValueTypeInfo()->name(), "vector<Vec3d>");
    EXPECT_EQ(dataVectorColor.getValueTypeInfo()->name(), "vector<RGBAColor>");
}
}// namespace sofa
