/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/vectorData.h>

#include <gtest/gtest.h>

namespace sofa {

/**  Test suite for data link.
Create two datas and a link between them.
Set the value of data1 and check if the boolean is dirty of data2 is true and that the value of data2 is right.
  */
struct DataLink_test: public ::testing::Test
{
    core::objectmodel::Data<int> data1;
    core::objectmodel::Data<int> data2;

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

// Test
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
    core::objectmodel::Data<int> data1;
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

// Test
TEST_F(vectorData_test , test_resize )
{
    this->test_resize();
}
TEST_F(vectorData_test , test_link )
{
    this->test_link();
}

}// namespace sofa
