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

#include "Binding_Vector.h"
#include <sofa/defaulttype/Vec.h>
#include "PythonToSofa.inl"

using sofa::defaulttype::Vector3 ;

static inline Vector3* get_vector3(PyObject* obj) {
    return sofa::py::unwrap<Vector3>(obj);
}


SP_CLASS_ATTR_GET(Vector3,x)(PyObject *self, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->x());
}


SP_CLASS_ATTR_SET(Vector3,x)(PyObject *self, PyObject * args, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->x()=PyFloat_AsDouble(args);
    return 0;
}


SP_CLASS_ATTR_GET(Vector3,y)(PyObject *self, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->y());
}


SP_CLASS_ATTR_SET(Vector3,y)(PyObject *self, PyObject * args, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->y()=PyFloat_AsDouble(args);
    return 0;
}


SP_CLASS_ATTR_GET(Vector3,z)(PyObject *self, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }
    return PyFloat_FromDouble(obj->z());
}


SP_CLASS_ATTR_SET(Vector3,z)(PyObject *self, PyObject * args, void*)
{
    Vector3* obj = get_vector3( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->z()=PyFloat_AsDouble(args);
    return 0;
}


// =============================================================================
// (de)allocator
// =============================================================================
PyObject * Vector3_PyNew(PyTypeObject * /*type*/, PyObject *args, PyObject * /*kwds*/)
{
    Vector3 *obj = new Vector3();
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
        return 0;
    obj->x()=x;
    obj->y()=y;
    obj->z()=z;
    return SP_BUILD_PYPTR(Vector3,Vector3,obj,true); // "true", because I manage the deletion myself (below)
}

void Vector3_PyFree(void * self)
{
    if (!((PyPtr<Vector3>*)self)->deletable) return;
    Vector3* obj = get_vector3( (PyObject*)self );
    delete obj; // done!
}



SP_CLASS_METHODS_BEGIN(Vector3)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(Vector3)
SP_CLASS_ATTR(Vector3,x)
SP_CLASS_ATTR(Vector3,y)
SP_CLASS_ATTR(Vector3,z)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(Vector3,Vector3)
