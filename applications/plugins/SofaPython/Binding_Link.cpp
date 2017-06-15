/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "Binding_Link.h"

#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/objectmodel/Link.h>


using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;


// TODO:
// se servir du LinkTypeInfo pour utiliser directement les bons type :-)
// Il y a un seul type "Link" exposé en python, le transtypage est géré automatiquement


SP_CLASS_ATTR_GET(Link,name)(PyObject *self, void*)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object; // TODO: check dynamic cast
    return PyString_FromString(link->getName().c_str());
}
SP_CLASS_ATTR_SET(Link,name)(PyObject *self, PyObject * args, void*)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object; // TODO: check dynamic cast
    char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
    link->setName(str);
    return 0;
}

PyObject *GetLinkValuePython(BaseLink* link)
{
    // only by string for now
    return PyString_FromString(link->getValueString().c_str());
}

int SetLinkValuePython(BaseLink* link, PyObject* args)
{
    // only by string for now

    // de quel type est args ?
    if( PyString_Check(args) )
    {
        // it's a string
        char *str = PyString_AsString(args); // for setters, only one object and not a tuple....
        link->read(str);
        return 0;
    }

    return -1;
}



SP_CLASS_ATTR_GET(Link,value)(PyObject *self, void*)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object; // TODO: check dynamic cast
    return GetLinkValuePython(link);
}

SP_CLASS_ATTR_SET(Link,value)(PyObject *self, PyObject * args, void*)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object; // TODO: check dynamic cast
    return SetLinkValuePython(link,args);
}

//// access ONE element of the vector
//extern "C" PyObject * Link_getValue(PyObject *self, PyObject * args)
//{
//    BaseLink* link=((PyPtr<BaseLink>*)self)->object;

//    int index;
//    if (!PyArg_ParseTuple(args, "i",&index))
//    {
//        PyErr_BadArgument();
//        Py_RETURN_NONE;
//    }

//    if ((size_t)index >= link->getSize())
//    {
//        // out of bounds!
//        SP_MESSAGE_ERROR( "Link.getValue index overflow" )
//        PyErr_BadArgument();
//        Py_RETURN_NONE;
//    }
//    if (typeinfo->Scalar())
//        return PyFloat_FromDouble(typeinfo->getScalarValue(link->getValueVoidPtr(),index));
//    if (typeinfo->Integer())
//        return PyInt_FromLong((long)typeinfo->getIntegerValue(link->getValueVoidPtr(),index));
//    if (typeinfo->Text())
//        return PyString_FromString(typeinfo->getTextValue(link->getValueVoidPtr(),index).c_str());

//    // should never happen....
//    SP_MESSAGE_ERROR( "Link.getValue unknown link type" )
//    PyErr_BadArgument();
//    Py_RETURN_NONE;
//}

//extern "C" PyObject * Link_setValue(PyObject *self, PyObject * args)
//{
//    BaseLink* link=((PyPtr<BaseLink>*)self)->object;
//    const AbstractTypeInfo *typeinfo = link->getValueTypeInfo(); // info about the link value
//    int index;
//    PyObject *value;
//    if (!PyArg_ParseTuple(args, "iO",&index,&value))
//    {
//        PyErr_BadArgument();
//        Py_RETURN_NONE;
//    }
//    if ((unsigned int)index>=typeinfo->size())
//    {
//        // out of bounds!
//        SP_MESSAGE_ERROR( "Link.setValue index overflow" )
//        PyErr_BadArgument();
//        Py_RETURN_NONE;
//    }
//    if (typeinfo->Scalar() && PyFloat_Check(value))
//    {
//        typeinfo->setScalarValue((void*)link->getValueVoidPtr(),index,PyFloat_AsDouble(value));
//        return PyInt_FromLong(0);
//    }
//    if (typeinfo->Integer() && PyInt_Check(value))
//    {
//        typeinfo->setIntegerValue((void*)link->getValueVoidPtr(),index,PyInt_AsLong(value));
//        return PyInt_FromLong(0);
//    }
//    if (typeinfo->Text() && PyString_Check(value))
//    {
//        typeinfo->setTextValue((void*)link->getValueVoidPtr(),index,PyString_AsString(value));
//        return PyInt_FromLong(0);
//    }

//    // should never happen....
//    SP_MESSAGE_ERROR( "Link.setValue type mismatch" )
//    PyErr_BadArgument();
//    Py_RETURN_NONE;
//}


extern "C" PyObject * Link_getValueTypeString(PyObject *self, PyObject * /*args*/)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object;
    return PyString_FromString(link->getValueTypeString().c_str());
}

extern "C" PyObject * Link_getValueString(PyObject *self, PyObject * /*args*/)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object;
    return PyString_FromString(link->getValueString().c_str());
}

extern "C" PyObject * Link_getSize(PyObject *self, PyObject * /*args*/)
{
    BaseLink* link=((PyPtr<BaseLink>*)self)->object;
    return PyInt_FromLong( link->getSize() );
}



SP_CLASS_METHODS_BEGIN(Link)
SP_CLASS_METHOD(Link,getValueTypeString)
SP_CLASS_METHOD(Link,getValueString)
//SP_CLASS_METHOD(Link,setValue)
//SP_CLASS_METHOD(Link,getValue)
SP_CLASS_METHOD(Link,getSize)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(Link)
SP_CLASS_ATTR(Link,name)
//SP_CLASS_ATTR(BaseLink,owner)
SP_CLASS_ATTR(Link,value)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR(Link,BaseLink)

