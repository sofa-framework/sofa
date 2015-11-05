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

#include "Binding_BaseMapping.h"
#include "Binding_BaseObject.h"

#include <sofa/core/BaseMapping.h>
using namespace sofa;
using namespace sofa::core;
using namespace sofa::core::objectmodel;



extern "C" PyObject * BaseMapping_getFrom(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping = dynamic_cast<BaseMapping*>(((PySPtr<Base>*)self)->object.get());

    helper::vector<BaseState*> from = mapping->getFrom();

    PyObject *list = PyList_New(from.size());

    for (unsigned int i=0; i<from.size(); ++i)
        PyList_SetItem(list,i,SP_BUILD_PYSPTR(from[i]));

    return list;
}

extern "C" PyObject * BaseMapping_getTo(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping = dynamic_cast<BaseMapping*>(((PySPtr<Base>*)self)->object.get());

    helper::vector<BaseState*> to = mapping->getTo();

    PyObject *list = PyList_New(to.size());

    for (unsigned int i=0; i<to.size(); ++i)
        PyList_SetItem(list,i,SP_BUILD_PYSPTR(to[i]));

    return list;
}



extern "C" PyObject * BaseMapping_setFrom(PyObject * self, PyObject * args)
{
    BaseMapping* mapping = dynamic_cast<BaseMapping*>(((PySPtr<Base>*)self)->object.get());

    PyObject* pyFrom;
    if (!PyArg_ParseTuple(args, "O",&pyFrom))
    {
        SP_MESSAGE_ERROR( "BaseMapping_setFrom: a BaseState* is required" );
        Py_RETURN_NONE;
    }

    BaseState* from=dynamic_cast<BaseState*>(((PySPtr<Base>*)pyFrom)->object.get());
    if (!from)
    {
        SP_MESSAGE_ERROR( "BaseMapping_setFrom: is not a BaseState*" );
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    mapping->setFrom( from );

    Py_RETURN_NONE;
}

extern "C" PyObject * BaseMapping_setTo(PyObject * self, PyObject * args)
{
    BaseMapping* mapping = dynamic_cast<BaseMapping*>(((PySPtr<Base>*)self)->object.get());

    PyObject* pyTo;
    if (!PyArg_ParseTuple(args, "O",&pyTo))
        Py_RETURN_NONE;

    BaseState* to=dynamic_cast<BaseState*>(((PySPtr<Base>*)pyTo)->object.get());
    if (!to)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    mapping->setTo( to );

    Py_RETURN_NONE;
}


SP_CLASS_METHODS_BEGIN(BaseMapping)
SP_CLASS_METHOD(BaseMapping,getFrom)
SP_CLASS_METHOD(BaseMapping,getTo)
SP_CLASS_METHOD(BaseMapping,setFrom)
SP_CLASS_METHOD(BaseMapping,setTo)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(BaseMapping,BaseMapping,BaseObject)


