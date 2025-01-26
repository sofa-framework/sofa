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
#include <sofa/core/DataEngine.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;


namespace sofa {


/// to test tracked Data
class TestEngine : public core::DataEngine
{

public:

    SOFA_CLASS(TestEngine,core::DataEngine);


    Data< bool > input;
    Data< int > output;

    enum { UNDEFINED=0, CHANGED, NO_CHANGED };

    TestEngine()
        : Inherit1()
        , input(initData(&input,false,"input","input"))
        , output(initData(&output,(int)UNDEFINED,"output","output"))
    {
        addInput(&input);
        addOutput(&output);
    }

    ~TestEngine() override {}

    void init() override
    {
        setDirtyValue();
    }

    void reinit() override
    {
        update();
    }

    void doUpdate() override
    {
        // true only iff the DataTracker associated to the Data 'input' is Dirty
        // that could only happen if 'input' was dirtied since last update
        if( m_dataTracker.hasChanged( input ) )
            output.setValue(CHANGED);
        else
            output.setValue(NO_CHANGED);
    }

};



struct DataEngine_test: public BaseTest
{
    TestEngine engine;

    void doSetUp() override
    {
        engine.init();
    }

    /// to test tracked Data
    template < class T >
    void testTrackedData(T& engine)
    {
        // input did not change, it is not dirtied, so neither its associated DataTracker
        ASSERT_EQ(engine.output.getValue(), TestEngine::NO_CHANGED);

        // modifying input sets it as dirty, so its associated DataTracker too
        engine.input.setValue(true);
        ASSERT_EQ(engine.output.getValue(), TestEngine::CHANGED);

        // nothing changes, no one is dirty
        engine.update();
        ASSERT_EQ(engine.output.getValue(), TestEngine::NO_CHANGED);

        // modifying input sets it as dirty, so its associated DataTracker too
        engine.input.setValue(true);
        // cleaning/accessing the input will not clean its associated DataTracker
        engine.input.cleanDirty();
        ASSERT_EQ(engine.output.getValue(), TestEngine::CHANGED);
    }

};

// Test
TEST_F(DataEngine_test, testDataEngine )
{
    this->testTrackedData<TestEngine>(this->engine);
}

}// namespace sofa
