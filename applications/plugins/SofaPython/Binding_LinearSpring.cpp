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

#include "Binding_LinearSpring.h"
#include <SofaDeformable/SpringForceField.h>
using namespace sofa::component::interactionforcefield;

extern "C" PyObject * LinearSpring_getAttr_Index1(PyObject *self, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    return PyInt_FromLong(obj->m1);
}

extern "C" int LinearSpring_setAttr_Index1(PyObject *self, PyObject * args, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
//    printf("***** DBG LinearSpring_setAttr_Index1 %d\n",(int)PyInt_AsLong(args));
    obj->m1=PyInt_AsLong(args);
    return 0;
}

extern "C" PyObject * LinearSpring_getAttr_Index2(PyObject *self, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyInt_FromLong(obj->m2);
}

extern "C" int LinearSpring_setAttr_Index2(PyObject *self, PyObject * args, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
//    printf("***** DBG LinearSpring_setAttr_Index2 %d\n",(int)PyInt_AsLong(args));
    obj->m2=PyInt_AsLong(args);
    return 0;
}

extern "C" PyObject * LinearSpring_getAttr_Ks(PyObject *self, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->ks);
}

extern "C" int LinearSpring_setAttr_Ks(PyObject *self, PyObject * args, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
//    printf("***** DBG LinearSpring_setAttr_Ks %f\n",(float)PyFloat_AsDouble(args));
    obj->ks=PyFloat_AsDouble(args);
    return 0;
}


extern "C" PyObject * LinearSpring_getAttr_Kd(PyObject *self, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->kd);
}

extern "C" int LinearSpring_setAttr_Kd(PyObject *self, PyObject * args, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
//    printf("***** DBG LinearSpring_setAttr_Kd %f\n",(float)PyFloat_AsDouble(args));
    obj->kd=PyFloat_AsDouble(args);
    return 0;
}


extern "C" PyObject * LinearSpring_getAttr_L(PyObject *self, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyFloat_FromDouble(obj->initpos);
}

extern "C" int LinearSpring_setAttr_L(PyObject *self, PyObject * args, void*)
{
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
//    printf("***** DBG LinearSpring_setAttr_L %f\n",(float)PyFloat_AsDouble(args));
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
    LinearSpring<SReal> *obj = new LinearSpring<SReal>(Index1,Index2,Ks,Kd,L);
    return SP_BUILD_PYPTR(LinearSpring,LinearSpring<SReal>,obj,true); // "true", because I manage the deletion myself (below)
}
void LinearSpring_PyFree(void * self)
{
    if (!((PyPtr<LinearSpring<SReal> >*)self)->deletable) return;
    LinearSpring<SReal>* obj=dynamic_cast<LinearSpring<SReal>*>(((PyPtr<LinearSpring<SReal> >*)self)->object);
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

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(LinearSpring,LinearSpring<SReal>)
