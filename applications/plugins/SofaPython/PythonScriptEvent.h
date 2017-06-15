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
#ifndef PYTHONSCRIPTEVENT_H
#define PYTHONSCRIPTEVENT_H

#include "PythonCommon.h"

#include <SofaPython/config.h>
#include "ScriptEvent.h"


namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 * @brief This event notifies about GUI interaction.
 */
class SOFA_SOFAPYTHON_API PythonScriptEvent : public sofa::core::objectmodel::ScriptEvent
{
public:

    SOFA_EVENT_H( PythonScriptEvent )

    /**
     * @brief Constructor.
     */
    PythonScriptEvent(sofa::simulation::Node::SPtr sender, const char* eventName, PyObject* userData);

    /**
     * @brief Destructor.
     */
    virtual ~PythonScriptEvent();

    /**
     * @brief Get the event name
     */
    PyObject* getUserData(void) const {return m_userData;}

    virtual const char* getClassName() const { return "PythonScriptEvent"; }
private:

    PyObject* m_userData;

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // PYTHONSCRIPTEVENT_H
