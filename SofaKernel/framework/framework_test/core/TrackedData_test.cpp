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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/DataTracker.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <gtest/gtest.h>

namespace sofa {


/// To test tracked Data
/// The 'input' Data is tracked.
/// At each time step, we can check if it changed.
/// If so, we can update 'depend_on_input' Data.
/// E.g. if the young modulus Data changed, we can update the sitffness matrix.
class TestObject : public core::objectmodel::BaseObject
{

public:

    SOFA_CLASS(TestObject,core::objectmodel::BaseObject);


    Data< bool > input;
    Data< int > depend_on_input;

    enum { UNDEFINED=0, CHANGED, NO_CHANGED };

    TestObject()
        : Inherit1()
        , input(initData(&input,false,"input","input"))
        , depend_on_input(initData(&depend_on_input,(int)UNDEFINED,"depend_on_input","depend_on_input"))
    {
        // we will check at each step, if the Data 'input' changed
        // note that it could be done anywhere else
        this->f_listening.setValue( true );

        // to track the Data  and be able to check if it changed
        m_dataTracker.trackData(input);
    }

    ~TestObject() {}


    // my own function to update Data
    void updateData()
    {
        // true only iff the DataTracker associated to the Data 'input' is Dirty
        // that could only happen if 'input' was dirtied since last update
        if( m_dataTracker.isDirty( input ) )
        {
            depend_on_input.setValue(CHANGED);
            m_dataTracker.clean( input );
        }
        else
            depend_on_input.setValue(NO_CHANGED);
    }



    void handleEvent( core::objectmodel::Event* _event )
    {
        if (simulation::AnimateBeginEvent::checkEventType(_event))
        {
            updateData(); // check if Data changed since last step
        }
    }

protected:

    core::DataTracker m_dataTracker;

};


struct DataTracker_test: public ::testing::Test
{
    TestObject testObject;

    void SetUp()
    {
        testObject.init();
    }

    /// to test tracked Data
    void testTrackedData()
    {
        // input did not change, it is not dirtied, so neither its associated DataTracker
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::NO_CHANGED);

        // modifying input sets it as dirty, so its associated DataTracker too
        testObject.input.setValue(true);
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::CHANGED);

        testObject.input.setValue(false);
        testObject.input.cleanDirty();
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::CHANGED);
    }

};

// Test
TEST_F(DataTracker_test, testTrackedData )
{
    this->testTrackedData();
}


//////////////////////



/// An example to automatically update 'depend_on_input'
/// when needed and when 'input' changed.
class TestObject2 : public core::objectmodel::BaseObject
{

public:

    SOFA_CLASS(TestObject2,core::objectmodel::BaseObject);


    Data< bool > input;
    Data< int > depend_on_input;

    enum { UNDEFINED=0, CHANGED, NO_CHANGED };

    TestObject2()
        : Inherit1()
        , input(initData(&input,false,"input","input"))
        , depend_on_input(initData(&depend_on_input,(int)UNDEFINED,"depend_on_input","depend_on_input"))
    {
        m_dataTracker.addInput(&input);
        m_dataTracker.addOutput(&depend_on_input);
    }

    ~TestObject2() {}


    // my own function to update Data
    void updateData()
    {
        // true only iff the DataTracker associated to the Data 'input' is Dirty
        // that could only happen if 'input' was dirtied since last update
        if( m_dataTracker.isDirty( input ) )
        {
            depend_on_input.setValue(CHANGED);
            m_dataTracker.clean( input );
        }
        else
            depend_on_input.setValue(NO_CHANGED);
    }



    void handleEvent( core::objectmodel::Event* _event )
    {
        if (simulation::AnimateBeginEvent::checkEventType(_event))
        {
            updateData(); // check if Data changed since last step
        }
    }

protected:

    core::DataTrackerDDGNode m_dataTracker;

};


struct DataTracker_test: public ::testing::Test
{
    TestObject testObject;

    void SetUp()
    {
        testObject.init();
    }

    /// to test tracked Data
    void testTrackedData()
    {
        // input did not change, it is not dirtied, so neither its associated DataTracker
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::NO_CHANGED);

        // modifying input sets it as dirty, so its associated DataTracker too
        testObject.input.setValue(true);
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::CHANGED);

        testObject.input.setValue(false);
        testObject.input.cleanDirty();
        testObject.updateData();
        ASSERT_TRUE(testObject.depend_on_input.getValue()==TestObject::CHANGED);
    }

};

// Test
TEST_F(DataTracker_test, testTrackedData )
{
    this->testTrackedData();
}


}// namespace sofa
