/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "PythonScriptFunction.h"

#include <SofaPython/PythonEnvironment.h>

namespace sofa
{

namespace core
{


namespace objectmodel
{

PythonScriptFunctionParameter::PythonScriptFunctionParameter() : ScriptFunctionParameter(),
	m_own(false),
	m_pyData(0)
{

}

PythonScriptFunctionParameter::PythonScriptFunctionParameter(PyObject* data, bool own) : ScriptFunctionParameter(),
	m_own(own),
	m_pyData(data)
{
    
}

PythonScriptFunctionParameter::~PythonScriptFunctionParameter()
{

}

PythonScriptFunctionResult::PythonScriptFunctionResult() : ScriptFunctionResult(),
	m_own(false),
	m_pyData(0)
{
	if(m_own && m_pyData) {
        simulation::PythonEnvironment::gil lock(__func__);
		Py_DECREF(m_pyData);
    }
}

PythonScriptFunctionResult::PythonScriptFunctionResult(PyObject* data, bool own) : ScriptFunctionResult(),
	m_own(own),
	m_pyData(data)
{

}

PythonScriptFunctionResult::~PythonScriptFunctionResult()
{
	if(m_own && m_pyData) {
        simulation::PythonEnvironment::gil lock(__func__);
		Py_DECREF(m_pyData);
    }
}

PythonScriptFunction::PythonScriptFunction(PyObject* pyCallableObject, bool own) : ScriptFunction(),
	m_own(own),
	m_pyCallableObject(pyCallableObject)
{
	
}

PythonScriptFunction::~PythonScriptFunction()
{
	if(m_own && m_pyCallableObject) {
        simulation::PythonEnvironment::gil lock(__func__);        
		Py_DECREF(m_pyCallableObject);
    }
}

void PythonScriptFunction::setCallableObject(PyObject* callableObject, bool own)
{
    if(m_pyCallableObject && m_own ) Py_DECREF(m_pyCallableObject);

    m_pyCallableObject = callableObject;
    m_own=own;
}

void PythonScriptFunction::onCall(const ScriptFunctionParameter* parameter, ScriptFunctionResult* result) const
{
	if(!m_pyCallableObject) {
		return;
    }

    simulation::PythonEnvironment::gil lock(__func__);
    
	const PythonScriptFunctionParameter* pythonScriptParameter = dynamic_cast<const PythonScriptFunctionParameter*>(parameter);
	PythonScriptFunctionResult* pythonScriptResult = dynamic_cast<PythonScriptFunctionResult*>(result);

	PyObject* pyParameter = 0;
	if(pythonScriptParameter) {
		pyParameter = pythonScriptParameter->data();
    }
    
	PyObject* pyResult = PyObject_CallObject(m_pyCallableObject, pyParameter);
	if(!pyResult) {
		SP_MESSAGE_EXCEPTION("in PythonScriptFunction: python function call failed")
		PyErr_Print();
	} else if(pythonScriptResult) {
		pythonScriptResult->setData(pyResult, true);
    }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa



