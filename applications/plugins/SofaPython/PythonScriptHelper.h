/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef PYTHONSCRIPTHELPER_H
#define PYTHONSCRIPTHELPER_H

#include <vector>

#include "PythonMacros.h"
#include <SofaPython/config.h>

#include <sofa/simulation/common/Simulation.h>

#include "PythonScriptController.h"
#include "PythonScriptFunction.h"

namespace sofa {
namespace helper {


namespace { // anonymous namespase

PyObject* PythonScript_valueToPyObject(bool param)
{
    PyObject* value = nullptr;
    value = Py_BuildValue("b", param);
    return value;
}
PyObject* PythonScript_valueToPyObject(int param)
{
    PyObject* value = nullptr;
    value = Py_BuildValue("i", param);
    return value;
}
PyObject* PythonScript_valueToPyObject(unsigned int param)
{
    PyObject* value = nullptr;
    value = Py_BuildValue("UI", param);
    return value;
}
PyObject* PythonScript_valueToPyObject(double param)
{
    PyObject* value = nullptr;
    value = Py_BuildValue("d", param);
    return value;
}
PyObject* PythonScript_valueToPyObject(std::string const& param)
{
    PyObject* value = nullptr;
    value = Py_BuildValue("s", param.c_str());
    return value;
}

void PythonScript_pyObjectToValue(PyObject* pyObject, bool & val)
{
    if (!pyObject) return;
    if(PyBool_Check(pyObject))
        val = (Py_False != pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to bool");
}

void PythonScript_pyObjectToValue(PyObject* pyObject, int & val)
{
    if (!pyObject) return;
    if(PyInt_Check(pyObject))
        val = (int)PyInt_AS_LONG(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to int");
}

void PythonScript_pyObjectToValue(PyObject* pyObject, double & val)
{
    if (!pyObject) return;
    if(PyFloat_Check(pyObject))
        val = PyFloat_AS_DOUBLE(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to double");
}


void PythonScript_parameterVector(std::vector<PyObject*> & /*vecParam*/) {return;}

template<typename T, typename... ParametersType>
void PythonScript_parameterVector(std::vector<PyObject*> & vecParam, T param, ParametersType... otherParameters)
{
    vecParam.push_back(PythonScript_valueToPyObject(param));
    PythonScript_parameterVector(vecParam, otherParameters...);
}

template<typename... ParametersType>
PyObject* PythonScript_parameterTuple(ParametersType... parameters)
{
    std::vector<PyObject*> vecParam;
    PythonScript_parameterVector(vecParam, parameters...);
    PyObject* tuple = PyTuple_New(vecParam.size());
    for (std::size_t i=0; i<vecParam.size(); ++i)
        PyTuple_SetItem(tuple, i, vecParam[i]);
    return tuple;
}

} // anonymous namespase

/// A helper function to call \a funcName in \a pythonScriptControllerName
template<typename ResultType, typename... ParametersType>
void PythonScriptFunction_call(ResultType & result, std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
{
    sofa::component::controller::PythonScriptController* controller = nullptr;
    controller = dynamic_cast<sofa::component::controller::PythonScriptController*>(sofa::simulation::getSimulation()->GetRoot()->getObject(pythonScriptControllerName.c_str()));
    if(!controller) {
        SP_MESSAGE_ERROR("Controller not found " << "(name: " << pythonScriptControllerName << " function: " << funcName << ")");
        return;
    }
    PyObject* pyCallableObject = PyObject_GetAttrString(controller->scriptControllerInstance(), funcName.c_str());
    if(!pyCallableObject) {
        SP_MESSAGE_ERROR("Function not found " << "(name: " << pythonScriptControllerName << " function: " << funcName << ")");
        return;
    }

    sofa::core::objectmodel::PythonScriptFunction pyFunction(pyCallableObject, true);
    sofa::core::objectmodel::PythonScriptFunctionParameter pyParameter(PythonScript_parameterTuple(parameters...), true);
    sofa::core::objectmodel::PythonScriptFunctionResult pyResult;
    pyFunction(&pyParameter, &pyResult);
    PythonScript_pyObjectToValue(pyResult.data(), result);
}

template<typename... ParametersType>
void PythonScriptFunction_callNoResult(std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
{
    int result; // dummy result
    PythonScriptFunction_call(result, pythonScriptControllerName, funcName, parameters...);
}

} // namespace helper
} // namespace sofa

#endif // PYTHONSCRIPTHELPER_H
