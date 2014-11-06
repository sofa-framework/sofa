/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_TestEngine_H
#define SOFA_COMPONENT_ENGINE_TestEngine_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/SofaGeneral.h>

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
class SOFA_ENGINE_API TestEngine : public core::DataEngine
{

protected:

    TestEngine();

    virtual ~TestEngine() {}
public:
    SOFA_CLASS(TestEngine,core::DataEngine);
    void init();

    void reinit();

    void update();

    // To see how many times update function is called
    int getCounterUpdate();

    void printUpdateCallList();

    Data<SReal> f_numberToMultiply;    ///< number to multiply
    Data<SReal> f_factor;  ///< multiplication factor
    Data<SReal> f_result;       ///< result

    int counter;

    int identifier;

    static int instance;

    static std::list<int> updateCallList;

};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
