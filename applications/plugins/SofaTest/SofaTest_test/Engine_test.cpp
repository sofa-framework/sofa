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
#include <sofa/component/engine/TestEngine.h>
#include "Sofa_test.h"


namespace sofa {

/**  Test suite for engine using TestEngine class.
This class has a counter which shows how many times the update method is called. 
Add inputs to engine.
Check that the output is updated only if necessary.
Check that update method is called only 1 time
  */
struct Engine_test : public Sofa_test<>
{
    typedef sofa::component::engine::TestEngine TestEngine;
    TestEngine::SPtr engine1;
    TestEngine::SPtr engine2;

    /// Create the engines
    void SetUp()
    { 
       // Engine 1
       engine1 = sofa::core::objectmodel::New<TestEngine>();
       engine1->f_numberToMultiply.setValue(1);
       engine1->f_factor.setValue(2);
       engine1->init();
        
       // Engine 2 linked to the ouput of engine 1
       engine2 = sofa::core::objectmodel::New<TestEngine>();
       sofa::modeling::setDataLink(&engine1->f_result,&engine2->f_numberToMultiply);
       engine2->f_factor.setValue(3);
       engine2->init();

    }

    // Test if the output is updated only if necessary
    void testUpdate()
    {
       //std::cout << "************Get Value Ouput E2*******************"<< std::endl;
       SReal result2 = engine2->f_result.getValue();

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

       // Test if result is correct
       ASSERT_EQ(result2,6);

    }

};

// first test case
TEST_F(Engine_test , check_engine_update )
{
    this->testUpdate();
}

}// namespace sofa