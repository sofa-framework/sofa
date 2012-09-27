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
#include <sofa/core/ObjectFactory.h>
using namespace sofa::core;
#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;

#include "Binding_BaseContext.h"
#include "Binding_Base.h"
#include "Binding_Vector.h"

#include <sofa/simulation/common/Node.h>
using namespace sofa::simulation;


extern "C" PyObject * BaseContext_setGravity(PyObject *self, PyObject * args)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    PyPtr<Vector3>* pyVec;
    if (!PyArg_ParseTuple(args, "O",&pyVec))
        return 0;
    obj->setGravity(*pyVec->object);
    return Py_BuildValue("i",0);
}

extern "C" PyObject * BaseContext_getGravity(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return SP_BUILD_PYPTR(Vector3,Vector3,new Vector3(obj->getGravity()),true); // "true", because I manage the deletion myself
}

extern "C" PyObject * BaseContext_getTime(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return PyFloat_FromDouble(obj->getTime());
}

extern "C" PyObject * BaseContext_getDt(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return PyFloat_FromDouble(obj->getDt());
}

extern "C" PyObject * BaseContext_getRootContext(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    return SP_BUILD_PYSPTR(obj->getRootContext());
}

// object factory
extern "C" PyObject * BaseContext_createObject(PyObject * self, PyObject * args)
{
    BaseContext* context=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    PyObject* pyDesc;
    if (!PyArg_ParseTuple(args, "O",&pyDesc))
        return 0;
    BaseObjectDescription *desc=(((PyPtr<BaseObjectDescription>*)pyDesc)->object);

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,desc);//.get();
    if (obj==0)
    {
        printf("<SofaPython> ERROR createObject '%s' of type '%s' in node '%s'\n",
                desc->getName().c_str(),
                desc->getAttribute("type",""),
                context->getName().c_str());
        PyErr_BadArgument();
        return 0;
    }

    Node *node = dynamic_cast<Node*>(context);
    if (node)
        node->init(sofa::core::ExecParams::defaultInstance());
    else
        obj->init();

    return SP_BUILD_PYSPTR(obj.get());
}


extern "C" PyObject * BaseContext_getObject(PyObject * self, PyObject * args)
{
    BaseContext* context=dynamic_cast<BaseContext*>(((PySPtr<Base>*)self)->object.get());
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
        return 0;
    if (!context || !path)
    {
        PyErr_BadArgument();
        return 0;
    }
    BaseObject::SPtr sptr;
    context->get<BaseObject>(sptr,path);

    return SP_BUILD_PYSPTR(sptr.get());
}

SP_CLASS_METHODS_BEGIN(BaseContext)
SP_CLASS_METHOD(BaseContext,getRootContext)
SP_CLASS_METHOD(BaseContext,getTime)
SP_CLASS_METHOD(BaseContext,getDt)
SP_CLASS_METHOD(BaseContext,getGravity)
SP_CLASS_METHOD(BaseContext,setGravity)
SP_CLASS_METHOD(BaseContext,createObject)
SP_CLASS_METHOD(BaseContext,getObject)
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

SP_CLASS_TYPE_SPTR_ATTR(BaseContext,BaseContext,Base)
