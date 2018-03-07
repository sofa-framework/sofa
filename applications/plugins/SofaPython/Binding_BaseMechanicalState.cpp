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

#include "Binding_BaseMechanicalState.h"
#include "Binding_BaseState.h"
#include "PythonToSofa.inl"

using sofa::core::behavior::BaseMechanicalState ;

static BaseMechanicalState* get_basemechanicalstate(PyObject* self) {
    return sofa::py::unwrap<BaseMechanicalState>(self);
}


static PyObject * BaseMechanicalState_applyTranslation(PyObject *self, PyObject * args)
{
    BaseMechanicalState* obj  = get_basemechanicalstate( self );
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; /// helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            return NULL;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyTranslation(dx,dy,dz);
    Py_RETURN_NONE;
}

static PyObject * BaseMechanicalState_applyScale(PyObject *self, PyObject * args)
{
    BaseMechanicalState* obj  = get_basemechanicalstate( self );
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; /// helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            return NULL;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyScale(dx,dy,dz);
    Py_RETURN_NONE;
}

static PyObject * BaseMechanicalState_applyRotation(PyObject *self, PyObject * args)
{
    BaseMechanicalState* obj  = get_basemechanicalstate( self );
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; /// helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            return NULL;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyRotation(dx,dy,dz);
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(BaseMechanicalState)
SP_CLASS_METHOD(BaseMechanicalState,applyTranslation)
SP_CLASS_METHOD(BaseMechanicalState,applyScale)
SP_CLASS_METHOD(BaseMechanicalState,applyRotation)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(BaseMechanicalState,BaseMechanicalState,BaseState)



