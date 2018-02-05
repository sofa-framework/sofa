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

#include "Binding_BaseState.h"
#include "Binding_BaseObject.h"
#include "PythonToSofa.inl"

using sofa::core::BaseState;

static BaseState* get_basestate(PyObject* self) {
    return sofa::py::unwrap<BaseState>(self);
}


static PyObject * BaseState_resize(PyObject *self, PyObject * args)
{
    BaseState* obj = get_basestate( self );
    int newSize;
    if (!PyArg_ParseTuple(args, "i", &newSize)) {
        return NULL;
    }

    obj->resize(newSize);
    Py_RETURN_NONE;
}


static PyObject * BaseState_getSize(PyObject *self, PyObject * args)
{
    BaseState* obj = get_basestate( self );

    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    return PyInt_FromSize_t(obj->getSize());
}


SP_CLASS_METHODS_BEGIN(BaseState)
SP_CLASS_METHOD(BaseState, resize)
SP_CLASS_METHOD(BaseState, getSize)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(BaseState, BaseState, BaseObject)
