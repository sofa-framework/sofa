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

#include "Binding_BaseObject.h"
#include "Binding_Base.h"
#include "PythonFactory.h"
#include "PythonToSofa.inl"

using sofa::core::objectmodel::BaseObject;

static BaseObject* get_baseobject(PyObject* self) {
    return sofa::py::unwrap<BaseObject>(self);
}


static PyObject * BaseObject_init(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->init();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_bwdInit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->bwdInit();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_reinit(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->reinit();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_storeResetState(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->storeResetState();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_reset(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->reset();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_cleanup(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    obj->cleanup();
    Py_RETURN_NONE;
}

static PyObject * BaseObject_getContext(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    return sofa::PythonFactory::toPython(obj->getContext());
}

static PyObject * BaseObject_getMaster(PyObject *self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );
    return sofa::PythonFactory::toPython(obj->getMaster());
}


static PyObject * BaseObject_setSrc(PyObject *self, PyObject * args)
{
    BaseObject* obj = get_baseobject( self );
    char *valueString;
    PyObject *pyLoader;
    if (!PyArg_ParseTuple(args, "sO",&valueString,&pyLoader)) {
        return NULL;
    }
    BaseObject* loader = get_baseobject( self );
    obj->setSrc(valueString,loader);
    Py_RETURN_NONE;
}

static PyObject * BaseObject_getPathName(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString(obj->getPathName().c_str());
}

// the same as 'getPathName' with a extra prefix '@'
static PyObject * BaseObject_getLinkPath(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString(("@"+obj->getPathName()).c_str());
}


static PyObject * BaseObject_getSlaves(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    const BaseObject::VecSlaves& slaves = obj->getSlaves();

    PyObject *list = PyList_New(slaves.size());

    for (unsigned int i=0; i<slaves.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(slaves[i].get()));

    return list;
}

static PyObject * BaseObject_getName(PyObject * self, PyObject * /*args*/)
{
    BaseObject* obj = get_baseobject( self );

    return PyString_FromString((obj->getName()).c_str());
}

extern "C" PyObject * BaseObject_getAsACreateObjectParameter(PyObject * self, PyObject *args)
{
    return BaseObject_getLinkPath(self, args);
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
SP_CLASS_METHOD(BaseObject,getSlaves)
SP_CLASS_METHOD(BaseObject,getName)
SP_CLASS_METHOD(BaseObject,getAsACreateObjectParameter)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(BaseObject,BaseObject,Base)
