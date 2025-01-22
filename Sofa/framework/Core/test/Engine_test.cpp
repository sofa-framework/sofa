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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/defaulttype/VecTypes.h>

#include "TestEngine.h"

namespace sofa {

/**  Test suite for engine using TestEngine class.
This class has a counter which shows how many times the update method is called.
Add inputs to engine.
Check that the output is updated only if necessary.
For this test 3 engines are used.The output of engine1 is linked to the input of engine2 and also the input of engine3.
         engine2
        /
engine1
        \
         engine3
  */
struct Engine_test : public BaseTest
{
    typedef sofa::component::engine::TestEngine TestEngine;
    TestEngine::SPtr engine1;
    TestEngine::SPtr engine2;
    TestEngine::SPtr engine3;

    /// Create the engines
    void doSetUp() override
    {
       // Engine 1
       engine1 = sofa::core::objectmodel::New<TestEngine>();
       engine1->f_numberToMultiply.setValue(1);
       engine1->f_factor.setValue(2);
       engine1->init();

       // Engine 2 linked to the output of engine 1
       engine2 = sofa::core::objectmodel::New<TestEngine>();
       engine2->f_numberToMultiply.setParent(&engine1->f_result);
       engine2->f_factor.setValue(3);
       engine2->init();

       // Engine 3 linked to the output of engine 1
       engine3 = sofa::core::objectmodel::New<TestEngine>();
       engine3->f_numberToMultiply.setParent(&engine1->f_result);
       engine3->f_factor.setValue(3);
       engine3->init();

    }

    // Test if the output of engine2 is updated only if necessary
    void testUpdateEngine2()
    {
        //Get output engine2
       SReal result2 = engine2->f_result.getValue();
       result2 = engine2->f_result.getValue();

       // Test if update method of engine1 is called 1 time
       if(engine1->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine1 was called " << engine1->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine2 is called 1 time
       if(engine2->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine2 was called " << engine2->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine3 is not called
       if(engine3->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine3 must not be called " << std::endl;
       }

       // Test if result is correct
       ASSERT_EQ(result2,6);

    }

    // Test if the output of engine3 is updated only if necessary
    void testUpdateEngine3()
    {
        //Get output engine3
        const SReal result3 = engine3->f_result.getValue();

       // Test if update method of engine1 is called 1 time
       if(engine1->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine1 was called " << engine1->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if update method of engine2 is not called
       if(engine2->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine2 must not be called " << std::endl;
       }

       // Test if update method of engine3 is called 1 time
       if(engine3->getCounterUpdate()!=1)
       {
           ADD_FAILURE() << "Update method of engine3 was called " << engine3->getCounterUpdate() << " times instead of 1 time " << std::endl;
       }

       // Test if result is correct
       ASSERT_EQ(result3,6);

    }

    // Test the propagation: if the output is changed the input must not changed
    void testPropagationDirection()
    {
        // Check propagation direction

       // Change output value of engine3
       engine3->f_result.setValue(2);

       // Check that update methods are not called

       if(engine1->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine1 must not be called " << std::endl;
       }

       if(engine2->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine2 must not be called " << std::endl;
       }

       if(engine3->getCounterUpdate()!=0)
       {
           ADD_FAILURE() << "Update method of engine3 must not be called " << std::endl;
       }

       // Check that input value is not changed
        const SReal input1 = engine1->f_numberToMultiply.getValue();

       ASSERT_EQ(input1,1);

    }

};

/// first test case: Check update method of engine2
TEST_F(Engine_test , check_engine2_update )
{
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testUpdateEngine2();
}

/// second test case: Check update method of engine3
TEST_F(Engine_test , check_engine3_update )
{
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testUpdateEngine3();
}

/// third test case: check propagation direction
TEST_F(Engine_test , check_propagation )
{
    EXPECT_MSG_NOEMIT(Error, Warning);
    this->testPropagationDirection();
}


}// namespace sofa
