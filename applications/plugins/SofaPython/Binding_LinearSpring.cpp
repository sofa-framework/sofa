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

#include "Binding_LinearSpring.h"
#include "PythonToSofa.inl"

using sofa::component::interactionforcefield::LinearSpring ;
typedef LinearSpring<SReal> LinearSpringR;


/// getting a LinearSpringR* from a PyObject*
static inline LinearSpringR* get_LinearSpringR(PyObject* obj) {
    return sofa::py::unwrap<LinearSpringR>(obj);
}


static PyObject * LinearSpring_getAttr_Index1(PyObject *self, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return NULL;
    }
    return PyInt_FromLong(obj->m1);
}


static int LinearSpring_setAttr_Index1(PyObject *self, PyObject * args, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return -1;
    }
    obj->m1=PyInt_AsLong(args);
    return 0;
}


static PyObject * LinearSpring_getAttr_Index2(PyObject *self, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return NULL;
    }
    return PyInt_FromLong(obj->m2);
}


static int LinearSpring_setAttr_Index2(PyObject *self, PyObject * args, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return -1;
    }
    obj->m2=PyInt_AsLong(args);
    return 0;
}


static PyObject * LinearSpring_getAttr_Ks(PyObject *self, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return NULL;
    }
    return PyFloat_FromDouble(obj->ks);
}


static int LinearSpring_setAttr_Ks(PyObject *self, PyObject * args, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return -1;
    }
    obj->ks=PyFloat_AsDouble(args);
    return 0;
}


static PyObject * LinearSpring_getAttr_Kd(PyObject *self, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return NULL;
    }
    return PyFloat_FromDouble(obj->kd);
}

static int LinearSpring_setAttr_Kd(PyObject *self, PyObject * args, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return -1;
    }
    obj->kd=PyFloat_AsDouble(args);
    return 0;
}


static PyObject * LinearSpring_getAttr_L(PyObject *self, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return NULL;
    }
    return PyFloat_FromDouble(obj->initpos);
}

static int LinearSpring_setAttr_L(PyObject *self, PyObject * args, void*)
{
    LinearSpringR* obj = get_LinearSpringR( self );
    if (!obj)
    {
        SP_PYERR_SETSTRING_INVALIDTYPE("LinearSpring<SReal>");
        return -1;
    }
    obj->initpos=PyFloat_AsDouble(args);
    return 0;
}



// =============================================================================
// (de)allocator
// =============================================================================
PyObject * LinearSpring_PyNew(PyTypeObject * /*type*/, PyObject *args, PyObject * /*kwds*/)
{
    int Index1,Index2;
    double Ks,Kd,L;
    if (!PyArg_ParseTuple(args, "iiddd",&Index1,&Index2,&Ks,&Kd,&L))
        return 0;
    LinearSpringR *obj = new LinearSpringR(Index1,Index2,Ks,Kd,L);
    return SP_BUILD_PYPTR(LinearSpring,LinearSpringR,obj,true); // "true", because I manage the deletion myself (below)
}
void LinearSpring_PyFree(void * self)
{
    if (!((PyPtr<LinearSpringR >*)self)->deletable) return;
    LinearSpringR* obj = get_LinearSpringR( (PyObject*)self );
    delete obj; // done!
}



SP_CLASS_METHODS_BEGIN(LinearSpring)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(LinearSpring)
SP_CLASS_ATTR(LinearSpring,Index1)
SP_CLASS_ATTR(LinearSpring,Index2)
SP_CLASS_ATTR(LinearSpring,Ks)
SP_CLASS_ATTR(LinearSpring,Kd)
SP_CLASS_ATTR(LinearSpring,L)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(LinearSpring,LinearSpringR)
