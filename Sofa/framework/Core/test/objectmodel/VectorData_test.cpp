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
using sofa::testing::BaseTest ;

namespace sofa {

struct VectorData_test: public ::testing::Test
{
    Data<int> data1;
    core::objectmodel::vectorData<int> vDataInt;

    VectorData_test()
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

TEST_F(VectorData_test , test_resize )
{
    this->test_resize();
}
TEST_F(VectorData_test , test_link )
{
    this->test_link();
}

} // namespace sofa
