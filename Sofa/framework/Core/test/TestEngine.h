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
#ifndef SOFA_COMPONENT_ENGINE_TestEngine_H
#define SOFA_COMPONENT_ENGINE_TestEngine_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace core::behavior;
using namespace core::topology;
using namespace core::objectmodel;

/**
 * This class is only used to test engine. 
 */
class TestEngine : public core::DataEngine
{

protected:

    TestEngine();

    ~TestEngine() override {}
public:
    SOFA_CLASS(TestEngine,core::DataEngine);
    void init() override;

    void reinit() override;

    void doUpdate() override;

    // To see how many times update function is called
    int getCounterUpdate();

    void printUpdateCallList();

    Data<SReal> f_numberToMultiply; ///< number that will be multiplied by the factor
    Data<SReal> f_factor; ///< multiplication factor
    Data<SReal> f_result; ///< result of the multiplication of numberToMultiply by factor

    int counter;

    int identifier;

    static int instance;

    static std::list<int> updateCallList;

};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
