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

#include <sofa/core/objectmodel/DataFileNameVector.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

namespace sofa {

using namespace core::objectmodel;

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
    dataFileNameVector.setValueAsString( "['"+std::string(SOFA_TESTING_RESOURCES_DIR) + "/dir with spaces/file.txt' ,'"+ std::string(SOFA_TESTING_RESOURCES_DIR) + "/file with spaces.txt' ]" );
    ASSERT_EQ( dataFileNameVector.getValue().size(), 2u );
}

TEST_F(DataFileNameVector_test , read_spaces )
{
    dataFileNameVector.read( "['" + std::string(SOFA_TESTING_RESOURCES_DIR) + "/dir with spaces/file.txt' ,'"+ std::string(SOFA_TESTING_RESOURCES_DIR) + "/file with spaces.txt' ]" );
    ASSERT_EQ( dataFileNameVector.getValue().size(), 2u );
}


}
