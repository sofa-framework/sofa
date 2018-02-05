/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_STANDARDTEST_DataEngine_test_H
#define SOFA_STANDARDTEST_DataEngine_test_H

#include "Sofa_test.h"

#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa {


/// @internal a DataEngine templated (and derived) from a given DataEngine
/// call defined DataEngine while counting the number of calls to the update function
template <typename DataEngineType>
class TestDataEngine : public DataEngineType
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(TestDataEngine,DataEngineType),DataEngineType);

    TestDataEngine()
        : DataEngineType()
        , m_counter( 0 )
    {}


    virtual void update() override
    {
        DataEngineType::update();

        ++m_counter;
    }

    void resetCounter() { m_counter = 0; }
    unsigned getCounter() const { return m_counter; }

private:

    unsigned m_counter; /// nb of call to update

};





/** @brief Helper for writing DataEngine tests.
 * @author Matthieu Nesme, 2015
 *
 */
template <typename DataEngineType>
struct DataEngine_test : public Sofa_test<>
{
    typedef TestDataEngine<DataEngineType> Engine;
    typedef core::objectmodel::DDGNode DDGNode;
    typedef DDGNode::DDGLinkContainer DDGLinkContainer;

    typename Engine::SPtr m_engine; ///< the real tested engine
    typename DataEngineType::SPtr m_engineInput; ///< an other identical engine, where only inputs are used (not the engine itself). It is an easy way to create all inputs of the right type, to be able to link wuth them.

    ///
    DataEngine_test()
    {
        m_engine = sofa::core::objectmodel::New<Engine>();
        m_engine->name.setValue("engine");
        m_engineInput = sofa::core::objectmodel::New<DataEngineType>();
        m_engineInput->name.setValue("engineInput");
    }

    virtual void init()
    {
        m_engineInput->init();
        m_engine->init();

        // linking tested engine's input with others data, just to be sure getting their value will inplicate a chain of updates
        const DDGLinkContainer& engine_inputs = m_engine->DDGNode::getInputs();
        const DDGLinkContainer& parent_inputs = m_engineInput->DDGNode::getInputs();
        for( unsigned i=0, iend=engine_inputs.size() ; i<iend ; ++i )
            static_cast<core::objectmodel::BaseData*>(engine_inputs[i])->setParent(static_cast<core::objectmodel::BaseData*>(parent_inputs[i]));
    }



// Note it is implemented as a macro so the error line number is better
#if 0
    #define CHECKCOUNTER( value ) ASSERT_TRUE( m_engine->getCounter() == value );
    #define CHECKMAXCOUNTER( value ) ASSERT_TRUE( m_engine->getCounter() <= value );
#else
    #define CHECKCOUNTER( value ) if( m_engine->getCounter() != value ) { ADD_FAILURE() << "Counter == "<<m_engine->getCounter()<<" != "<<value<<std::endl; ASSERT_TRUE( m_engine->getCounter() == value ); }
    #define CHECKMAXCOUNTER( value ) if( m_engine->getCounter() > value ) { ADD_FAILURE() << "Counter == "<<m_engine->getCounter()<<" > "<<value<<std::endl; ASSERT_TRUE( m_engine->getCounter() <= value ); }
#endif

    /// Testing the number of call to the DataEngine::update() function
    /// @warning DO NOT test the values computed by the engine
    /// To do so, you can inherit this class and add a test function that takes inputs and ouputs to test
    void run_basic_test()
    {
        /// The comp
        {
            IGNORE_MSG(Error) ;
            init();
        }

        m_engine->resetCounter();

        const DDGLinkContainer& inputs = m_engine->DDGNode::getInputs();

        CHECKCOUNTER( 0 );  // c'est parti mon kiki
        const DDGLinkContainer& parent_inputs = m_engineInput->DDGNode::getInputs();


        CHECKCOUNTER( 0 );  // c'est parti mon kiki
        const DDGLinkContainer& outputs = m_engine->DDGNode::getOutputs();


        CHECKCOUNTER( 0 );  // c'est parti mon kiki

        // modifying inputs to ensure the engine should be evaluated
        for( unsigned i=0, iend=parent_inputs.size() ; i<iend ; ++i )
        {
            parent_inputs[i]->setDirtyValue();
            CHECKCOUNTER( 0 );  // c'est parti mon kiki
        }

        CHECKCOUNTER( 0 );  // c'est parti mon kiki

        outputs[0]->updateIfDirty(); // could call the engine
        CHECKMAXCOUNTER( 1 );

        m_engine->resetCounter();
        outputs[0]->updateIfDirty(); // should not call the engine
        CHECKCOUNTER( 0 );


        // modifying the parent inputs one by one
        m_engine->resetCounter();
        for( unsigned i=0, iend=parent_inputs.size() ; i<iend ; ++i )
        {
            parent_inputs[i]->setDirtyValue(); // to check if the engine is evaluated for this input
            outputs[0]->updateIfDirty(); // could call the engine
        }
        CHECKMAXCOUNTER( parent_inputs.size() );

        // modifying the engine inputs one by one
        m_engine->resetCounter();
        for( unsigned i=0, iend=inputs.size() ; i<iend ; ++i )
        {
            inputs[i]->setDirtyValue(); // to check if the engine is evaluated for this input
            outputs[0]->updateIfDirty(); // could call the engine
        }
        CHECKMAXCOUNTER( parent_inputs.size() );
    }


};


} // namespace sofa

#endif
