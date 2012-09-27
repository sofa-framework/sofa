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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "Binding_MechanicalObject.h"
#include "Binding_BaseState.h"
#include "Binding_Vector.h"

#include <sofa/component/typedef/Sofa_typedef.h>

using namespace sofa::core::behavior;
using namespace sofa::core;



extern "C" PyObject * MechanicalObject_applyTranslation(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyTranslation(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_applyScale(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyScale(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_applyRotation(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->applyRotation(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_setTranslation(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->setTranslation(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_setScale(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->setScale(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_setRotation(PyObject *self, PyObject * args)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    double dx,dy,dz;
    if (!PyArg_ParseTuple(args, "ddd",&dx,&dy,&dz))
    {
        int ix,iy,iz; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iii",&ix,&iy,&iz))
        {
            PyErr_BadArgument();
            return 0;
        }
        dx = (double)ix;
        dy = (double)iy;
        dz = (double)iz;
    }
    obj->setRotation(dx,dy,dz);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * MechanicalObject_getTranslation(PyObject *self, PyObject * /*args*/)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    Vector3 *vec = new Vector3(obj->getTranslation());
    return SP_BUILD_PYPTR(Vector3,Vector3,vec,true); // "true", because I manage the deletion myself (below)
}

extern "C" PyObject * MechanicalObject_getRotation(PyObject *self, PyObject * /*args*/)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    Vector3 *vec = new Vector3(obj->getRotation());
    return SP_BUILD_PYPTR(Vector3,Vector3,vec,true); // "true", because I manage the deletion myself (below)
}

extern "C" PyObject * MechanicalObject_getScale(PyObject *self, PyObject * /*args*/)
{
    MechanicalObject3* obj=dynamic_cast<MechanicalObject3*>(((PySPtr<Base>*)self)->object.get());
    Vector3 *vec = new Vector3(obj->getScale());
    return SP_BUILD_PYPTR(Vector3,Vector3,vec,true); // "true", because I manage the deletion myself (below)
}


SP_CLASS_METHODS_BEGIN(MechanicalObject)
SP_CLASS_METHOD(MechanicalObject,setTranslation)
SP_CLASS_METHOD(MechanicalObject,setScale)
SP_CLASS_METHOD(MechanicalObject,setRotation)
SP_CLASS_METHOD(MechanicalObject,getTranslation)
SP_CLASS_METHOD(MechanicalObject,getScale)
SP_CLASS_METHOD(MechanicalObject,getRotation)
SP_CLASS_METHOD(MechanicalObject,applyTranslation)  // BaseMechanicalState func
SP_CLASS_METHOD(MechanicalObject,applyScale)        // BaseMechanicalState func
SP_CLASS_METHOD(MechanicalObject,applyRotation)     // BaseMechanicalState func
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(MechanicalObject,MechanicalObject3,BaseState)


