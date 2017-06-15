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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/DataTracker.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/helper/cast.h>

#include <gtest/gtest.h>

namespace sofa {


/// To test tracked Data
/// The 'input' Data is tracked.
/// At each time step, we can check if it changed.
/// If so, we can update 'depend_on_input' Data.
///
/// E.g. if the young modulus Data changed, we can update the sitffness matrix.
///
/// Cons: extra test at regular intervals to check if the Data changed.
/// Pros: it is clear what's going on (and when).
///       Maybe easier to manage multithreading?
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
/// when asking for its value and when 'input' changed.
/// It's the regular Data flow but w/o a sofa component (DataEngine).
/// The connected Data can be everywhere (inside a component, in between components)
/// not only between input and output of a DataEngine.
///
/// Pros: very flexible
/// Cons: more complex w/ multithreading??
///       It is not obvious to know who is an input, who is an output
class TestObject2 : public core::objectmodel::BaseObject
{

public:

    SOFA_CLASS(TestObject2,core::objectmodel::BaseObject);


    Data< bool > input;
    Data< bool > depend_on_input;


    TestObject2()
        : Inherit1()
        , input(initData(&input,false,"input","input"))
        , depend_on_input(initData(&depend_on_input,"depend_on_input","depend_on_input"))
    {
        m_dataTracker.addInput(&input); // several inputs can be added
        m_dataTracker.addOutput(&depend_on_input); // several output can be added
        m_dataTracker.setUpdateCallback( &TestObject2::myUpdate );
        m_dataTracker.setDirtyValue();
    }

    ~TestObject2() {}

    static unsigned s_updateCounter;


protected:

    core::DataTrackerEngine m_dataTracker;


    static void myUpdate( core::DataTrackerEngine* dataTrackerEngine )
    {
        ++s_updateCounter;

        // get the list of inputs for this DDGNode
        const core::DataTrackerEngine::DDGLinkContainer& inputs = dataTrackerEngine->getInputs();
        // get the list of outputs for this DDGNode
        const core::DataTrackerEngine::DDGLinkContainer& outputs = dataTrackerEngine->getOutputs();

        // we known who is who from the order Data were added to the DataTrackerEngine
        bool input = static_cast<Data< bool >*>( inputs[0] )->getValue();

        dataTrackerEngine->cleanDirty();


        Data< bool >* output = static_cast<Data< bool >*>( outputs[0] );

        output->setValue( input );
    }

};
unsigned TestObject2::s_updateCounter = 0u;

class DummyObject : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DummyObject,core::objectmodel::BaseObject);
    Data< bool > myData;
    DummyObject()
        : Inherit1()
        , myData(initData(&myData,false,"myData","myData"))
    {}
};



struct DataTrackerEngine_test: public ::testing::Test
{

    static unsigned updateCounter;
    void SetUp()
    {
        updateCounter = 0;
    }


    /// to test DataTrackerEngine between Data in the same component
    void testInsideComponent()
    {
        TestObject2 testObject;
        testObject.init();

        unsigned localCounter = 0u;

        ASSERT_TRUE(testObject.depend_on_input.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );

        testObject.input.setValue(true);
        ASSERT_TRUE(testObject.depend_on_input.getValue()==true);
        ++localCounter;
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );
        ASSERT_TRUE(testObject.depend_on_input.getValue()==true);
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );

        testObject.input.setValue(false);
        ASSERT_TRUE(testObject.depend_on_input.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );

        testObject.input.setValue(false);
        ASSERT_TRUE(testObject.depend_on_input.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );


        testObject.depend_on_input.setValue(true);
        ASSERT_TRUE(testObject.depend_on_input.getValue()==true);
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );

        testObject.input.setValue(false);
        ASSERT_TRUE(testObject.depend_on_input.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, TestObject2::s_updateCounter );
    }


    static void myUpdate( core::DataTrackerEngine* dataTrackerEngine )
    {
        ++updateCounter;

        // get the list of inputs for this DDGNode
        const core::DataTrackerEngine::DDGLinkContainer& inputs = dataTrackerEngine->getInputs();
        // get the list of outputs for this DDGNode
        const core::DataTrackerEngine::DDGLinkContainer& outputs = dataTrackerEngine->getOutputs();

        // we known who is who from the order Data were added to the DataTrackerEngine
        bool input = static_cast<Data< bool >*>( inputs[0] )->getValue();

        dataTrackerEngine->cleanDirty();


        Data< bool >* output = static_cast<Data< bool >*>( outputs[0] );

        output->setValue( input );
    }


    /// to test DataTrackerEngine between Data in separated components
    void testBetweenComponents()
    {


        DummyObject testObject, testObject2;

        core::DataTrackerEngine dataTracker;
        dataTracker.addInput(&testObject.myData); // several inputs can be added
        dataTracker.addOutput(&testObject2.myData); // several output can be added
        dataTracker.setUpdateCallback( &DataTrackerEngine_test::myUpdate );
        dataTracker.setDirtyValue();



        unsigned localCounter = 0u;

        testObject.myData.setValue(true);
        ASSERT_TRUE(testObject2.myData.getValue()==true);
        ++localCounter;
        ASSERT_EQ( localCounter, updateCounter );
        ASSERT_TRUE(testObject2.myData.getValue()==true);
        ASSERT_EQ( localCounter, updateCounter );

        testObject.myData.setValue(false);
        ASSERT_TRUE(testObject2.myData.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, updateCounter );

        testObject.myData.setValue(false);
        ASSERT_TRUE(testObject2.myData.getValue()==false);
        ++localCounter;
        ASSERT_EQ( localCounter, updateCounter );
    }

};
unsigned DataTrackerEngine_test::updateCounter = 0u;

// Test
TEST_F(DataTrackerEngine_test, testTrackedData )
{
    this->testInsideComponent();
    this->testBetweenComponents();
}

//////////////////////////////


/// Testing DataTrackerFunctor
/// to call a functor as soon as a Data is modified.
///
/// This mechanism is useful is some specific situations
/// such as updating particular fields in a gui.
/// An idea to do so: the functor can add the modified Data in a list
/// that will be read by the gui when it is refreshed in order to
/// update something.
struct DataTrackerFunctor_test: public ::testing::Test
{

    // This functor illustrates what is possible.
    // It shows how to get access to its inputs,
    // that it can have its own variables...
    struct MyDataFunctor
    {
        MyDataFunctor() : m_counter(0u) {}

        void operator() ( core::DataTrackerFunctor<MyDataFunctor>* tracker )
        {
            core::objectmodel::BaseData* data = down_cast<core::objectmodel::BaseData>( tracker->getInputs()[0] );
            msg_info("MyDataFunctor")<<"Data "<<data->getName()<<" just changed for the "<<++m_counter<<"-th time";
        }

        unsigned m_counter;
    };



    /// to test DataTrackerEngine between Data in the same component
    void test()
    {
        DummyObject testObject;
        testObject.init();


        // as soon as testObject.myData changes, myDataFunctor will be triggered
        MyDataFunctor myDataFunctor;
        core::DataTrackerFunctor<MyDataFunctor> myDataTrackerFunctor( myDataFunctor );
        myDataTrackerFunctor.addInput( &testObject.myData );

        // the functor is called when adding an input
        ASSERT_EQ( 1u, myDataFunctor.m_counter );




        // modifying the Data is calling the functor
        testObject.myData.setValue( false );
        ASSERT_EQ( 2u, myDataFunctor.m_counter );

        // getting the value is not calling the function
        testObject.myData.getValue();
        ASSERT_EQ( 2u, myDataFunctor.m_counter );

        // modifying the Data even with the same value is calling the functor
        // note it would be possible to do your own functor,
        // that keep a hash of the previous value
        // if you really need to check when the data trully changed
        testObject.myData.setValue( false );
        ASSERT_EQ( 3u, myDataFunctor.m_counter );

        // editing the data
        // the behavior would be simular with beginEdit()
        bool* t = testObject.myData.beginWriteOnly(); // the functor is triggered as soon as the edition begins
        ASSERT_EQ( 4u, myDataFunctor.m_counter );
        *t = true; // the functor is not triggered
        ASSERT_EQ( 4u, myDataFunctor.m_counter );
        *t = false; // the functor is not triggered
        ASSERT_EQ( 4u, myDataFunctor.m_counter );
        testObject.myData.endEdit();

        // example with an accessor
        {
        helper::WriteOnlyAccessor<core::objectmodel::Data<bool>> acc( testObject.myData );  // the functor is triggered as soon as the edition begins
        ASSERT_EQ( 5u, myDataFunctor.m_counter );
        *acc = false;  // the functor is not triggered
        *acc = true; // the functor is not triggered
        ASSERT_EQ( 5u, myDataFunctor.m_counter );
        }

    }
};


// Test
TEST_F(DataTrackerFunctor_test, test )
{
    this->test();
}





}// namespace sofa
