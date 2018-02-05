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

#include "PythonScriptControllerHelper.h"

namespace sofa {
namespace helper {

namespace internal {

PyObject* PythonScriptController_valueToPyObject(bool param)
{
    PyObject* value = NULL;
    value = Py_BuildValue("b", param);
    return value;
}
PyObject* PythonScriptController_valueToPyObject(int param)
{
    PyObject* value = NULL;
    value = Py_BuildValue("i", param);
    return value;
}
PyObject* PythonScriptController_valueToPyObject(unsigned int param)
{
    PyObject* value = NULL;
    value = Py_BuildValue("I", param);
    return value;
}
PyObject* PythonScriptController_valueToPyObject(double param)
{
    PyObject* value = NULL;
    value = Py_BuildValue("d", param);
    return value;
}
PyObject* PythonScriptController_valueToPyObject(std::string const& param)
{
    PyObject* value = NULL;
    value = Py_BuildValue("s", param.c_str());
    return value;
}

void PythonScriptController_pyObjectToValue(PyObject* pyObject, bool & val)
{
    if (pyObject && pyObject!=Py_None && PyBool_Check(pyObject))
        val = (Py_True == pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to bool");
}

void PythonScriptController_pyObjectToValue(PyObject* pyObject, int & val)
{
    if (pyObject && pyObject!=Py_None && PyInt_Check(pyObject))
        val = (int)PyInt_AS_LONG(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to int");
}
void PythonScriptController_pyObjectToValue(PyObject* pyObject, unsigned int & val)
{
    if (pyObject && pyObject!=Py_None && PyInt_Check(pyObject))
        val = (unsigned int)PyInt_AsUnsignedLongMask(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to unsigned int");
}
void PythonScriptController_pyObjectToValue(PyObject* pyObject, float & val)
{
    if (pyObject && pyObject!=Py_None && PyFloat_Check(pyObject))
        val = (float)PyFloat_AS_DOUBLE(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to float");
}

void PythonScriptController_pyObjectToValue(PyObject* pyObject, double & val)
{
    if (pyObject && pyObject!=Py_None && PyFloat_Check(pyObject))
        val = PyFloat_AS_DOUBLE(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to double");
}
void PythonScriptController_pyObjectToValue(PyObject* pyObject, std::string & val)
{
    if (pyObject && pyObject!=Py_None && PyString_Check(pyObject))
        val = PyString_AS_STRING(pyObject);
    else
        SP_MESSAGE_ERROR("Cannot convert pyObject to std::string");
}

}  // namespase internal
}  // namespace helper
} // namespace sofa
