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
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


namespace sofa {

using namespace core::objectmodel;

class Data_test : public BaseTest
{
public:
    Data<int> dataInt1;
    Data<int> dataInt2;
    Data<float> dataFloat;
    Data<double> dataDouble;
    Data<bool> dataBool;
    Data<sofa::defaulttype::Vec3> dataVec3;
    Data<sofa::helper::vector<sofa::defaulttype::Vec3>> dataVectorVec3;
    Data<sofa::helper::vector<sofa::helper::types::RGBAColor>> dataVectorColor;

    void SetUp() override
    {
        dataInt1.setName("dataInt1");
        dataInt2.setName("dataInt2");
    }

    void TearDown() override
    {
        dataInt1.unSetParent();
        dataInt2.unSetParent();
    }
};

TEST_F(Data_test, getValueTypeInfoEquality)
{
    EXPECT_EQ(dataInt1.getValueTypeInfo(), dataInt2.getValueTypeInfo());
    EXPECT_EQ(dataFloat.getValueTypeInfo(), Data<float>::GetValueTypeInfo());
}

TEST_F(Data_test, getValueTypeString)
{
    EXPECT_EQ(dataInt1.getValueTypeString(), "int");
    EXPECT_EQ(dataFloat.getValueTypeString(), "float");
    EXPECT_EQ(dataBool.getValueTypeString(), "bool");
    EXPECT_EQ(dataVec3.getValueTypeString(), "Vec3d");
    EXPECT_EQ(dataVectorVec3.getValueTypeString(), "vector<Vec3d>");
    EXPECT_EQ(dataVectorColor.getValueTypeString(), "vector<RGBAColor>");
}

TEST_F(Data_test, getNameWithValueTypeInfo)
{
    EXPECT_EQ(dataInt1.getValueTypeInfo()->name(), "int");
    EXPECT_EQ(dataFloat.getValueTypeInfo()->name(), "float");
    EXPECT_EQ(dataBool.getValueTypeInfo()->name(), "bool");
    EXPECT_EQ(dataVec3.getValueTypeInfo()->name(), "Vec3d");
    EXPECT_EQ(dataVectorVec3.getValueTypeInfo()->name(), "vector<Vec3d>");
    EXPECT_EQ(dataVectorColor.getValueTypeInfo()->name(), "vector<RGBAColor>");
}

TEST_F(Data_test, setValue)
{
    dataInt1.setValue(1);
    EXPECT_EQ(dataInt1.getValue(), 1);
    dataInt1.setValue(2);
    EXPECT_EQ(dataInt1.getValue(), 2);
}

TEST_F(Data_test, isDirty)
{
    EXPECT_FALSE(dataInt1.isDirty());
    dataInt1.setValue(1);
    EXPECT_FALSE(dataInt1.isDirty());
}

TEST_F(Data_test, getCounter)
{
    auto c = dataInt1.getCounter();
    dataInt1.setValue(1);
    EXPECT_EQ(c+1, dataInt1.getCounter());
}

TEST_F(Data_test, setParent)
{
    ASSERT_FALSE(dataInt1.hasParent());
    ASSERT_FALSE(dataInt2.hasParent());
    dataInt2.setParent(&dataInt1);
    ASSERT_FALSE(dataInt1.hasParent());
    ASSERT_TRUE(dataInt2.hasParent());
}

/// This test is checking if a parent data has its dirty flag correctly setted-up
/// when it is setted.
TEST_F(Data_test, setParentWithInit)
{
    dataInt1.setValue(1);
    dataInt2.setValue(0);
    dataInt2.setParent(&dataInt1);
    ASSERT_FALSE(dataInt1.isDirty());
    ASSERT_TRUE(dataInt2.isDirty());

    ASSERT_EQ(dataInt1.getValue(), 1);
    ASSERT_EQ(dataInt2.getValue(), 1);

    ASSERT_FALSE(dataInt1.isDirty());
    ASSERT_FALSE(dataInt2.isDirty());
}


TEST_F(Data_test, setValueWithParent)
{
    dataInt1.setValue(1);
    dataInt2.setValue(0);
    dataInt2.setParent(&dataInt1);

    dataInt2.setValue(-1);
    ASSERT_FALSE(dataInt2.hasParent());
    ASSERT_EQ(dataInt2.getValue(),-1);
    dataInt1.setValue(1);
    ASSERT_FALSE(dataInt2.hasParent());
    ASSERT_EQ(dataInt2.getValue(),-1);
}

} /// namespace sofa
