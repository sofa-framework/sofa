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
#ifndef PYTHONSCRIPTCONTROLLERHELPER_H
#define PYTHONSCRIPTCONTROLLERHELPER_H

#include "PythonMacros.h"

#include <vector>
#include <string>

#include <SofaPython/config.h>

#include <sofa/simulation/Simulation.h>

#include "PythonScriptController.h"
#include "PythonScriptFunction.h"

namespace sofa {
namespace helper {


namespace internal {

SOFA_SOFAPYTHON_API PyObject* PythonScriptController_valueToPyObject(bool param);
SOFA_SOFAPYTHON_API PyObject* PythonScriptController_valueToPyObject(int param);
SOFA_SOFAPYTHON_API PyObject* PythonScriptController_valueToPyObject(unsigned int param);
SOFA_SOFAPYTHON_API PyObject* PythonScriptController_valueToPyObject(double param);
SOFA_SOFAPYTHON_API PyObject* PythonScriptController_valueToPyObject(std::string const& param);

SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, bool & val);
SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, int & val);
SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, unsigned int & val);
SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, float & val);
SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, double & val);
SOFA_SOFAPYTHON_API void PythonScriptController_pyObjectToValue(PyObject* pyObject, std::string & val);


void PythonScriptController_parametersToVector(std::vector<PyObject*> & /*vecParam*/) {return;}

template<typename T, typename... ParametersType>
void PythonScriptController_parametersToVector(std::vector<PyObject*> & vecParam, T param, ParametersType... otherParameters)
{
    vecParam.push_back(PythonScriptController_valueToPyObject(param));
    PythonScriptController_parametersToVector(vecParam, otherParameters...);
}

template<typename... ParametersType>
PyObject* PythonScript_parametersToTuple(ParametersType... parameters)
{
    std::vector<PyObject*> vecParam;
    PythonScriptController_parametersToVector(vecParam, parameters...);
    PyObject* tuple = PyTuple_New(vecParam.size());
    for (std::size_t i=0; i<vecParam.size(); ++i)
        PyTuple_SetItem(tuple, i, vecParam[i]);
    return tuple;
}

} // namespase internal

/** A helper function to call \a funcName in \a pythonScriptControllerName.
 * The function returned value is stored in \a result.
 * If the controller functions returns \c None, or if you are not interested by the returned value, call it with \c nullptr as first parameter.
 */
template<typename ResultType, typename... ParametersType>
void PythonScriptController_call(ResultType * result, sofa::simulation::Node::SPtr root, std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
{
    sofa::component::controller::PythonScriptController* controller = nullptr;
    controller = dynamic_cast<sofa::component::controller::PythonScriptController*>(root->getObject(pythonScriptControllerName.c_str()));
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
    sofa::core::objectmodel::PythonScriptFunctionParameter pyParameter(internal::PythonScript_parametersToTuple(parameters...), true);
    sofa::core::objectmodel::PythonScriptFunctionResult pyResult;
    pyFunction(&pyParameter, &pyResult);
    if (result!=nullptr)
        internal::PythonScriptController_pyObjectToValue(pyResult.data(), *result);
}

template<typename... ParametersType>
void PythonScriptController_call(std::nullptr_t /*result*/, sofa::simulation::Node::SPtr root, std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
{
    int* none=nullptr;
    PythonScriptController_call(none, root, pythonScriptControllerName, funcName, parameters...);
}

} // namespace helper
} // namespace sofa

#endif // PYTHONSCRIPTCONTROLLERHELPER_H
