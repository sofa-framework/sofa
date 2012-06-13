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

#include "Binding_Vec3.h"

extern "C" PyObject * Vec3_getAttr_x(PyObject *self, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->x());
}
extern "C" int Vec3_setAttr_x(PyObject *self, PyObject * args, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->x()=PyFloat_AsDouble(args);
    return 0;
}

extern "C" PyObject * Vec3_getAttr_y(PyObject *self, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->y());
}
extern "C" int Vec3_setAttr_y(PyObject *self, PyObject * args, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->y()=PyFloat_AsDouble(args);
    return 0;
}

extern "C" PyObject * Vec3_getAttr_z(PyObject *self, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->z());
}
extern "C" int Vec3_setAttr_z(PyObject *self, PyObject * args, void*)
{
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->z()=PyFloat_AsDouble(args);
    return 0;
}


// =============================================================================
// (de)allocator
// =============================================================================
PyObject * Vec3_PyNew(PyTypeObject * /*type*/, PyObject *args, PyObject * /*kwds*/)
{
    Vec3 *obj = new Vec3();
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
        return 0;
    obj->x()=x;
    obj->y()=y;
    obj->z()=z;
    return SP_BUILD_PYPTR(Vec3,Vec3,obj,true); // "true", because I manage the deletion myself (below)
}
void Vec3_PyFree(void * self)
{
    if (!((PyPtr<Vec3>*)self)->deletable) return;
    Vec3* obj=dynamic_cast<Vec3*>(((PyPtr<Vec3>*)self)->object);
    delete obj; // done!
}



SP_CLASS_METHODS_BEGIN(Vec3)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(Vec3)
SP_CLASS_ATTR(Vec3,x)
SP_CLASS_ATTR(Vec3,y)
SP_CLASS_ATTR(Vec3,z)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(Vec3,Vec3)
