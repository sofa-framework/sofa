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
#include <sofa/helper/vectorData.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


namespace sofa {

using namespace core::objectmodel;

class Data_test : public BaseTest
{
public:
    Data<int> dataInt;
    Data<float> dataFloat;
    Data<bool> dataBool;
    Data<sofa::defaulttype::Vec3> dataVec3;
    Data<sofa::helper::vector<sofa::defaulttype::Vec3>> dataVectorVec3;
    Data<sofa::helper::vector<sofa::helper::types::RGBAColor>> dataVectorColor;
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

/** Test suite for vectorData
 *
 * @author Thomas Lemaire @date 2014
 */
struct vectorData_test: public ::testing::Test
{
    Data<int> data1;
    helper::vectorData<int> vDataInt;

    vectorData_test()
        : vDataInt(nullptr,"","")
    { }

    void SetUp() override
    {}

    void test_resize()
    {
        vDataInt.resize(3);
        ASSERT_EQ(vDataInt.size(),3u);
        vDataInt.resize(10);
        ASSERT_EQ(vDataInt.size(),10u);
        vDataInt.resize(8);
        ASSERT_EQ(vDataInt.size(),8u);
    }

    void test_link()
    {
        vDataInt.resize(5);
        vDataInt[3]->setParent(&data1);
        data1.setValue(1);
        ASSERT_NE(vDataInt[3]->getParent(),nullptr);
        ASSERT_EQ(vDataInt[3]->getValue(),1);
    }

};

TEST_F(vectorData_test , test_resize )
{
    this->test_resize();
}
TEST_F(vectorData_test , test_link )
{
    this->test_link();
}


/////////////////////////////////


/** Test suite for DataFileNameVector
 *
 * @author M Nesme @date 2016
 */
struct DataFileNameVector_test: public ::testing::Test
{
    DataFileNameVector dataFileNameVector;

    DataFileNameVector_test()
        : dataFileNameVector()
    { }

    void SetUp() override
    {}

};

TEST_F(DataFileNameVector_test , setValueAsString_spaces )
{
    dataFileNameVector.setValueAsString( "['"+std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir with spaces/file.txt' ,'"+ std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/file with spaces.txt' ]" );
    ASSERT_EQ( dataFileNameVector.getValue().size(), 2u );
}

TEST_F(DataFileNameVector_test , read_spaces )
{
    dataFileNameVector.read( "['" + std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir with spaces/file.txt' ,'"+ std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/file with spaces.txt' ]" );
    ASSERT_EQ( dataFileNameVector.getValue().size(), 2u );
}


}// namespace sofa
