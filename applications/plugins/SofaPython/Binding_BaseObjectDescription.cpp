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

#include "Binding_BaseObjectDescription.h"
#include "sofa/core/objectmodel/BaseObjectDescription.h"

using namespace sofa::core::objectmodel;


// =============================================================================
// attributes
// =============================================================================

extern "C" PyObject * BaseObjectDescription_getAttr_name(PyObject *self, void*)
{
    BaseObjectDescription* obj=dynamic_cast<BaseObjectDescription*>(((PyPtr<BaseObjectDescription>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    return PyString_FromString(obj->getName().c_str());
}
extern "C" int BaseObjectDescription_setAttr_name(PyObject *self, PyObject * args, void*)
{
    BaseObjectDescription* obj=dynamic_cast<BaseObjectDescription*>(((PyPtr<BaseObjectDescription>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->setName(PyString_AsString(args));
    return 0;
}


// =============================================================================
// methods
// =============================================================================

extern "C" PyObject * BaseObjectDescription_getAttribute(PyObject *self, PyObject * args)
{
    BaseObjectDescription* obj=dynamic_cast<BaseObjectDescription*>(((PyPtr<BaseObjectDescription>*)self)->object);
    char *argName;
    if (!PyArg_ParseTuple(args, "s",&argName))
        return 0;
    return PyString_FromString(obj->getAttribute(argName,""));
}

extern "C" PyObject * BaseObjectDescription_setAttribute(PyObject *self, PyObject * args)
{
    BaseObjectDescription* obj=dynamic_cast<BaseObjectDescription*>(((PyPtr<BaseObjectDescription>*)self)->object);
    char *argName;
    char *argValue;
    if (!PyArg_ParseTuple(args, "ss",&argName,&argValue))
        return 0;
    obj->setAttribute(argName,argValue);
    return Py_BuildValue("i",0);
}




// =============================================================================
// (de)allocator
// =============================================================================
PyObject * BaseObjectDescription_PyNew(PyTypeObject * /*type*/, PyObject *args, PyObject * /*kwds*/)
{
    char *name;
    char *type;
    if (!PyArg_ParseTuple(args, "ss",&name,&type))
        return 0;
    //printf("BaseObjectDescription name=%s type =%s\n",name,type);
    return SP_BUILD_PYPTR(BaseObjectDescription,BaseObjectDescription,new BaseObjectDescription(name,type),true); // "true", because I manage the deletion myself (below)
}
void BaseObjectDescription_PyFree(void * self)
{
    if (!((PyPtr<BaseObjectDescription>*)self)->deletable) return;
    BaseObjectDescription* obj=dynamic_cast<BaseObjectDescription*>(((PyPtr<BaseObjectDescription>*)self)->object);
    delete obj; // done!
}



SP_CLASS_METHODS_BEGIN(BaseObjectDescription)
SP_CLASS_METHOD(BaseObjectDescription,getAttribute)
SP_CLASS_METHOD(BaseObjectDescription,setAttribute)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(BaseObjectDescription)
SP_CLASS_ATTR(BaseObjectDescription,name)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(BaseObjectDescription,BaseObjectDescription)
