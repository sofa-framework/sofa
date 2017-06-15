/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <gtest/gtest.h>

namespace sofa {

using namespace core::objectmodel;

/**  Test suite for data link.
Create two datas and a link between them.
Set the value of data1 and check if the boolean is dirty of data2 is true and that the value of data2 is right.
  */
struct DataLink_test: public ::testing::Test
{
    Data<int> data1;
    Data<int> data2;

    /// Create a link between the two datas
    void SetUp()
    {
        // Link
        data2.setParent(&data1);
    }

    // Test if the output is updated only if necessary
    void testDataLink()
    {
        data1.setValue(1);
        ASSERT_FALSE(data1.isDirty());
        ASSERT_TRUE(data2.isDirty());
        ASSERT_TRUE(data2.getValue()!=0);

    }

};


/////////////////////////////////////


TEST_F(DataLink_test , testDataLink )
{
    this->testDataLink();
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
        : vDataInt(NULL,"","")
    { }

    void SetUp()
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

    void SetUp()
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
