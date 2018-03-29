/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_TestEngine_INL
#define SOFA_COMPONENT_ENGINE_TestEngine_INL

#include "TestEngine.h"

namespace sofa
{

namespace component
{

namespace engine
{

int TestEngine::instance = 0;
std::list<int> TestEngine::updateCallList;

using namespace core::behavior;
using namespace core::objectmodel;

TestEngine::TestEngine()
    : f_numberToMultiply( initData (&f_numberToMultiply, "number", "number that will be multiplied by the factor") )
    , f_factor(initData (&f_factor,"factor", "multiplication factor") )
    , f_result( initData (&f_result, "result", "result of the multiplication of numberToMultiply by factor") )
{
    counter = 0;
    instance++;
    this->identifier =  instance;
}

void TestEngine::init()
{
    addInput(&f_factor);
    addInput(&f_numberToMultiply);
    addOutput(&f_result);
    setDirtyValue();
}

void TestEngine::reinit()
{
    update();
}

void TestEngine::update()
{
    // Count how many times the update method is called
    counter ++;


///// FIRST get (and update) all read-only inputs

    // Get number to multiply
    SReal number = f_numberToMultiply.getValue(); 
    
    // Get factor
    SReal factor = f_factor.getValue();


///// THEN tell everthing is (will be) up to date now
/// @warning This must be done AFTER updating all inputs
/// can be done before or after setting up the outputs
    cleanDirty();


///// Compute all write-only outputs
    // Set result
    f_result.setValue(number*factor);
   
    // Update call list
    updateCallList.push_back(this->identifier);
}

int TestEngine::getCounterUpdate()
{
    return this->counter;
}

void TestEngine::printUpdateCallList()
{

    for (std::list<int>::iterator it=updateCallList.begin(); it != updateCallList.end(); ++it)
        std::cout << " Call engine " <<  *it <<std::endl;
   
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
