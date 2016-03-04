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
#ifndef PYTHONSCRIPTCONTROLLERHELPER_H
#define PYTHONSCRIPTCONTROLLERHELPER_H

#include "PythonMacros.h"

#include <vector>
#include <string>

#include <SofaPython/config.h>

#include <sofa/simulation/common/Simulation.h>

#include "PythonScriptController.h"
#include "PythonScriptFunction.h"

namespace sofa {
namespace helper {


namespace internal {

PyObject* PythonScriptController_valueToPyObject(bool param);
PyObject* PythonScriptController_valueToPyObject(int param);
PyObject* PythonScriptController_valueToPyObject(unsigned int param);
PyObject* PythonScriptController_valueToPyObject(double param);
PyObject* PythonScriptController_valueToPyObject(std::string const& param);

void PythonScriptController_pyObjectToValue(PyObject* pyObject, bool & val);
void PythonScriptController_pyObjectToValue(PyObject* pyObject, int & val);
void PythonScriptController_pyObjectToValue(PyObject* pyObject, unsigned int & val);
void PythonScriptController_pyObjectToValue(PyObject* pyObject, float & val);
void PythonScriptController_pyObjectToValue(PyObject* pyObject, double & val);
void PythonScriptController_pyObjectToValue(PyObject* pyObject, std::string & val);


void PythonScriptController_parametersToVector(std::vector<PyObject*> & /*vecParam*/) {return;}

#if __cplusplus > 201100L
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
#else
// implement a c++98 version if necessary
#endif

} // namespase internal

#if __cplusplus > 201100L
/** A helper function to call \a funcName in \a pythonScriptControllerName.
 * The function returned value is stored in \a result.
 * If the controller functions returns \c None, or if you are not interested by the returned value, call it with \c nullptr as first parameter.
 */
template<typename ResultType, typename... ParametersType>
void PythonScriptController_call(ResultType * result, std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
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
    sofa::core::objectmodel::PythonScriptFunctionParameter pyParameter(internal::PythonScript_parametersToTuple(parameters...), true);
    sofa::core::objectmodel::PythonScriptFunctionResult pyResult;
    pyFunction(&pyParameter, &pyResult);
    if (result!=nullptr)
        internal::PythonScriptController_pyObjectToValue(pyResult.data(), *result);
}

template<typename... ParametersType>
void PythonScriptController_call(std::nullptr_t /*result*/, std::string const& pythonScriptControllerName, std::string const& funcName, ParametersType... parameters)
{
    int* none=nullptr;
    PythonScriptController_call(none, pythonScriptControllerName, funcName, parameters...);
}

#else
// implement a c++98 version if necessary
#endif

} // namespace helper
} // namespace sofa

#endif // PYTHONSCRIPTCONTROLLERHELPER_H
