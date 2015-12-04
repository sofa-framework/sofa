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

#include "Binding_BaseObject.h"
#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;

extern "C" PyObject * BaseObject_init(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->init();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_bwdInit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->bwdInit();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_reinit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->reinit();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_storeResetState(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->storeResetState();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_reset(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->reset();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_cleanup(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->cleanup();
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_getContext(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    return SP_BUILD_PYSPTR(obj->getContext());
}

extern "C" PyObject * BaseObject_getMaster(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    return SP_BUILD_PYSPTR(obj->getMaster());
}


extern "C" PyObject * BaseObject_setSrc(PyObject *self, PyObject * args)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();
    char *valueString;
    PyObject *pyLoader;
    if (!PyArg_ParseTuple(args, "sO",&valueString,&pyLoader))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    BaseObject* loader=((PySPtr<Base>*)self)->object->toBaseObject();
    obj->setSrc(valueString,loader);
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseObject_getPathName(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();

    return PyString_FromString(obj->getPathName().c_str());
}

// the same as 'getPathName' with a extra prefix '@'
extern "C" PyObject * BaseObject_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj=((PySPtr<Base>*)self)->object->toBaseObject();

    return PyString_FromString(("@"+obj->getPathName()).c_str());
}


SP_CLASS_METHODS_BEGIN(BaseObject)
SP_CLASS_METHOD(BaseObject,init)
SP_CLASS_METHOD(BaseObject,bwdInit)
SP_CLASS_METHOD(BaseObject,reinit)
SP_CLASS_METHOD(BaseObject,storeResetState)
SP_CLASS_METHOD(BaseObject,reset)
SP_CLASS_METHOD(BaseObject,cleanup)
SP_CLASS_METHOD(BaseObject,getContext)
SP_CLASS_METHOD(BaseObject,getMaster)
SP_CLASS_METHOD(BaseObject,setSrc)
SP_CLASS_METHOD(BaseObject,getPathName)
SP_CLASS_METHOD(BaseObject,getLinkPath)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(BaseObject,BaseObject,Base)
