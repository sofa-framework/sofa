/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
using sofa::core::objectmodel::Data;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


/**  Test suite for data link.
Create two datas and a link between them.
Set the value of data1 and check if the boolean is dirty of data2 is true and that the value of data2 is right.
  */
struct DataLink_test: public BaseTest
{
    Data<int> data1;
    Data<int> data2;

    /// This method is defined in gtest framework to setting the test up.
    void SetUp() override
    {
        /// Setup the data and create a link between the two datas
        data1.setName("data1");
        data2.setName("data2");
    }

    void TearDown() override
    {
        data1.unset();
        data2.unset();
    }
};

/// This test check that the setting/unsetting mechanisme when the value is changed is working
/// Currently in Sofa the parenting link is not broken if the value is written.
TEST_F(DataLink_test, UnsetByValue)
{
    data2.setParent(&data1);
    ASSERT_NE(data2.getParent(), nullptr);
    data2.setValue(0);
    ASSERT_NE(data2.getParent(), nullptr);
}

/// This test check that the setting/unsetting mechanisme is working
TEST_F(DataLink_test, Set)
{
    ASSERT_EQ(data1.getParent(),nullptr);
    ASSERT_EQ(data2.getParent(),nullptr);
    data2.setParent(&data1);
    ASSERT_EQ(data1.getParent(),nullptr);
    ASSERT_EQ(data2.getParent(), &data1);
}

/// This test check that the setting/unsetting mechanisme is working
TEST_F(DataLink_test, Unset)
{
    ASSERT_EQ(data1.getParent(),nullptr);
    ASSERT_EQ(data2.getParent(),nullptr);
    data2.setParent(&data1);
    data2.setParent(nullptr);
    ASSERT_EQ(data1.getParent(),nullptr);
    ASSERT_EQ(data2.getParent(), nullptr);
}

TEST_F(DataLink_test, updateFromParent)
{
    data2.setParent(&data1);
    data1.setValue(1);
    ASSERT_FALSE(data1.isDirty());   ///< it is not dirty as the new value has been updated because of the set
    ASSERT_EQ(data1.getValue(), 1);
    ASSERT_NE(data2.getParent(),nullptr);
    ASSERT_TRUE(data2.isDirty());    ///< it is dirty as the parent's value has changed but there we no getValue, so update was not done
    ASSERT_EQ(data1.getValue(), data2.getValue());
}
