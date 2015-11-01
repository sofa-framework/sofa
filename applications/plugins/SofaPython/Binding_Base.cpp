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
#include "Binding_Base.h"
#include "Binding_Data.h"
#include "Binding_DisplayFlagsData.h"

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
using namespace sofa::core::objectmodel;
#include <sofa/core/visual/DisplayFlags.h>
using namespace sofa::core::visual;

extern "C" PyObject * Base_findData(PyObject *self, PyObject * args)
{
    Base* obj=dynamic_cast<Base*>(((PySPtr<Base>*)self)->object.get());
    char *dataName;
    if (!PyArg_ParseTuple(args, "s",&dataName))
        Py_RETURN_NONE;
    BaseData * data = obj->findData(dataName);
    if (!data)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    if (dynamic_cast<Data<DisplayFlags>*>(data))
        return SP_BUILD_PYPTR(DisplayFlagsData,BaseData,data,false);
    return SP_BUILD_PYPTR(Data,BaseData,data,false);
}

// Generic accessor to Data fields (in python native type)
extern "C" PyObject* Base_GetAttr(PyObject *o, PyObject *attr_name)
{
    Base* obj=dynamic_cast<Base*>(((PySPtr<Base>*)o)->object.get());
    char *dataName = PyString_AsString(attr_name);
//    printf("Base_GetAttr type=%s name=%s attrName=%s\n",obj->getClassName().c_str(),obj->getName().c_str(),dataName);

    // attribute does not exist: see if a Data field has this name...
    BaseData * data = obj->findData(dataName);
    if (!data)
    {
//        printf("Base_GetAttr ERROR data not found - type=%s name=%s attrName=%s\n",obj->getClassName().c_str(),obj->getName().c_str(),dataName);

        return PyObject_GenericGetAttr(o,attr_name);;
    }
    // we have our data... let's create the right Python type....
    return GetDataValuePython(data);
}

extern "C" int Base_SetAttr(PyObject *o, PyObject *attr_name, PyObject *v)
{
    // attribute does not exist: see if a Data field has this name...
    Base* obj=dynamic_cast<Base*>(((PySPtr<Base>*)o)->object.get());
    char *dataName = PyString_AsString(attr_name);
//    printf("Base_SetAttr name=%s\n",dataName);
    BaseData * data = obj->findData(dataName);
    if (!data)
        return PyObject_GenericSetAttr(o,attr_name,v);
    // we have our data... let's create the right Python type....
    if (!SetDataValuePython(data,v))
        return -1;
    return 0;
}

extern "C" PyObject * Base_getClassName(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not binded in SofaPython, so getPathName is binded in Node instead
    Base* node = dynamic_cast<Base*>(((PySPtr<Base>*)self)->object.get());

    return PyString_FromString(node->getClassName().c_str());
}

extern "C" PyObject * Base_getTemplateName(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not binded in SofaPython, so getPathName is binded in Node instead
    Base* node = dynamic_cast<Base*>(((PySPtr<Base>*)self)->object.get());

    return PyString_FromString(node->getTemplateName().c_str());
}

extern "C" PyObject * Base_getName(PyObject * self, PyObject * /*args*/)
{
    // BaseNode is not binded in SofaPython, so getPathName is binded in Node instead
    Base* node = dynamic_cast<Base*>(((PySPtr<Base>*)self)->object.get());

    return PyString_FromString(node->getName().c_str());
}

SP_CLASS_METHODS_BEGIN(Base)
SP_CLASS_METHOD(Base,findData)
SP_CLASS_METHOD(Base,getClassName)
SP_CLASS_METHOD(Base,getTemplateName)
SP_CLASS_METHOD(Base,getName)
SP_CLASS_METHODS_END


//SP_CLASS_DATA_ATTRIBUTE(Base,name)

SP_CLASS_ATTRS_BEGIN(Base)
//SP_CLASS_ATTR(Base,name)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_SPTR_ATTR_GETATTR(Base,Base)
