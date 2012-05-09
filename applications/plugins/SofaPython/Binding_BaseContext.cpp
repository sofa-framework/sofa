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
#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;

#include "Binding_BaseContext.h"
#include "Binding_Base.h"
#include "Binding_Vec3.h"



extern "C" PyObject * BaseContext_setGravity(PyObject *self, PyObject * args)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    PyPtr<Vec3>* pyVec;
    if (!PyArg_ParseTuple(args, "O",&pyVec))
        return 0;
    obj->setGravity(*pyVec->object);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * BaseContext_getGravity(PyObject *self, PyObject * args)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return SP_BUILD_PYPTR(Vec3,new Vec3(obj->getGravity()),true); // "true", because I manage the deletion myself
}


SP_CLASS_METHODS_BEGIN(BaseContext)
//SP_CLASS_METHOD(BaseContext,getRootContext)
//SP_CLASS_METHOD(BaseContext,getTime)
//SP_CLASS_METHOD(BaseContext,getDt)
SP_CLASS_METHOD(BaseContext,getGravity)
SP_CLASS_METHOD(BaseContext,setGravity)
SP_CLASS_METHODS_END


extern "C" PyObject * BaseContext_getAttr_animate(PyObject *self, void*)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return PyBool_FromLong(obj->getAnimate());
}
extern "C" int BaseContext_setAttr_animate(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    if (!PyBool_Check(args))
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->setAnimate(args==Py_True);
    return 0;
}

extern "C" PyObject * BaseContext_getAttr_active(PyObject *self, void*)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return PyBool_FromLong(obj->isActive());
}
extern "C" int BaseContext_setAttr_active(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    if (!PyBool_Check(args))
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->setActive(args==Py_True);
    return 0;
}


/*
extern "C" PyObject * BaseContext_getAttr_gravity(PyObject *self, void*)
{
    printf("BaseContext_getAttr_gravity\n");
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return SP_BUILD_PYPTR(Vec3,new Vec3(obj->getGravity()),true); // "true", because I manage the deletion myself
}
extern "C" int BaseContext_setAttr_gravity(PyObject *self, PyObject * args, void*)
{
    printf("BaseContext_setAttr_gravity\n");
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    PyPtr<Vec3>* pyVec;
    if (!PyArg_ParseTuple(args, "(O)",&pyVec))
    {
        printf("PyArg_ParseTuple error\n");
        printf("%s\n",PyString_AsString(args));
        return 0;
    }
    obj->setGravity(*pyVec->object);
    return 0;
}
*/

SP_CLASS_ATTRS_BEGIN(BaseContext)
SP_CLASS_ATTR(BaseContext,active)
SP_CLASS_ATTR(BaseContext,animate)
//SP_CLASS_ATTR(BaseContext,gravity) // attribut objets = problème... le setter ne fonctionne pas
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_SPTR_ATTR(BaseContext,Base)
