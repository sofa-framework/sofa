/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "stdafx.h"
#include <sofa/core/objectmodel/Data.h>
#include <plugins/SceneCreator/SceneCreator.h>
#include "Sofa_test.h"

namespace sofa {

/**  Test suite for data link.
Create two datas and a link between them.
Set the value of data1 and check if the boolean is dirty of data2 is true and that the value of data2 is right.
  */
struct DataLink_test : public Sofa_test<>
{
    core::objectmodel::Data<int> data1;
    core::objectmodel::Data<int> data2;

    /// Create a link between the two datas
    void SetUp()
    { 
       // Link
       sofa::modeling::setDataLink(&data1,&data2);

    }

    // Test if the output is updated only if necessary
    void testDataLink()
    {
       // Set the value of data1
        data1.setValue(1);

        // Test if boolean isDirty of data1 is false
        ASSERT_EQ(data1.isDirty(),0);

       // Test if boolean isDirty of data2 is true
       ASSERT_EQ(data2.isDirty(),1);

       // Test if result is correct
       ASSERT_EQ(data2.getValue(),1);

    }

};

// Test 
TEST_F(DataLink_test , test_update )
{
    this->testDataLink();
}

}// namespace sofa