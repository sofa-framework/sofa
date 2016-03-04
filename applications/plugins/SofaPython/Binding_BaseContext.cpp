/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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


#include "Binding_BaseContext.h"
#include "Binding_Base.h"
#include "Binding_Vector.h"
#include "ScriptEnvironment.h"

#include <sofa/defaulttype/Vec3Types.h>
using namespace sofa::defaulttype;
#include <sofa/core/ObjectFactory.h>
using namespace sofa::core;
#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;
#include <sofa/simulation/common/Node.h>
using namespace sofa::simulation;
using namespace sofa::defaulttype;


extern "C" PyObject * BaseContext_setGravity(PyObject *self, PyObject * args)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    PyPtr<Vector3>* pyVec;
    if (!PyArg_ParseTuple(args, "O",&pyVec))
        Py_RETURN_NONE;
    obj->setGravity(*pyVec->object);
    Py_RETURN_NONE;
}

extern "C" PyObject * BaseContext_getGravity(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return SP_BUILD_PYPTR(Vector3,Vector3,new Vector3(obj->getGravity()),true); // "true", because I manage the deletion myself
}

extern "C" PyObject * BaseContext_getTime(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return PyFloat_FromDouble(obj->getTime());
}

extern "C" PyObject * BaseContext_getDt(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return PyFloat_FromDouble(obj->getDt());
}

extern "C" PyObject * BaseContext_getRootContext(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return SP_BUILD_PYSPTR(obj->getRootContext());
}

// object factory
extern "C" PyObject * BaseContext_createObject_Impl(PyObject * self, PyObject * args, PyObject * kw, bool printWarnings)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();

//    std::cout << "<PYTHON> BaseContext_createObject PyTuple_Size=" << PyTuple_Size(args) << " PyDict_Size=" << PyDict_Size(kw) << std::endl;

    char *type;
    if (!PyArg_ParseTuple(args, "s",&type))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    // temporarily, the name is set to the type name.
    // if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type,type);

    if (kw && PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            PyObject *key = PyList_GetItem(keys,i);
            PyObject *value = PyList_GetItem(values,i);
        //    std::cout << PyString_AsString(PyList_GetItem(keys,i)) << "=\"" << PyString_AsString(PyObject_Str(PyList_GetItem(values,i))) << "\"" << std::endl;
            if (PyString_Check(value))
                desc.setAttribute(PyString_AsString(key),PyString_AsString(value));
            else
                desc.setAttribute(PyString_AsString(key),PyString_AsString(PyObject_Str(value)));
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,&desc);
    if (obj==0)
    {
        SP_MESSAGE_ERROR( "createObject: component '" << desc.getName() << "' of type '" << desc.getAttribute("type","")<< "' in node '"<<context->getName()<<"'" )
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    if( printWarnings )
    {
        Node *node = static_cast<Node*>(context);
        if (node)
        {
            //SP_MESSAGE_INFO( "Sofa.Node.createObject("<<type<<") node="<<node->getName()<<" isInitialized()="<<node->isInitialized() )
            if (node->isInitialized())
                SP_MESSAGE_WARNING( "Sofa.Node.createObject("<<type<<") called on a node("<<node->getName()<<") that is already initialized" )
    //        if (!ScriptEnvironment::isNodeCreatedByScript(node))
    //            SP_MESSAGE_WARNING( "Sofa.Node.createObject("<<type<<") called on a node("<<node->getName()<<") that is not created by the script" )
        }
    }

    return SP_BUILD_PYSPTR(obj.get());
}
extern "C" PyObject * BaseContext_createObject(PyObject * self, PyObject * args, PyObject * kw)
{
    return BaseContext_createObject_Impl( self, args, kw, true );
}
extern "C" PyObject * BaseContext_createObject_noWarning(PyObject * self, PyObject * args, PyObject * kw)
{
    return BaseContext_createObject_Impl( self, args, kw, false );
}

extern "C" PyObject * BaseContext_getObject(PyObject * self, PyObject * args)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
    {
        SP_MESSAGE_WARNING( "BaseContext_getObject: wrong argument, should be a string" )
        Py_RETURN_NONE;
    }
    if (!context || !path)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    BaseObject::SPtr sptr;
    context->get<BaseObject>(sptr,path);
    if (!sptr)
    {
        SP_MESSAGE_WARNING( "BaseContext_getObject: component "<<path<<" not found (the complete relative path is needed)" )
        Py_RETURN_NONE;
    }

    return SP_BUILD_PYSPTR(sptr.get());
}

extern "C" PyObject * BaseContext_getObjects(PyObject * self, PyObject * /*args*/)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();

    if (!context)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    BaseObject::SPtr sptr;

    sofa::helper::vector< boost::intrusive_ptr<BaseObject> > list;
    context->get<BaseObject>(&list,sofa::core::objectmodel::BaseContext::Local);

    PyObject *pyList = PyList_New(list.size());
    for (size_t i=0; i<list.size(); i++)
        PyList_SetItem(pyList, (Py_ssize_t)i, SP_BUILD_PYSPTR(list[i].get()));

    return pyList;
}

SP_CLASS_METHODS_BEGIN(BaseContext)
SP_CLASS_METHOD(BaseContext,getRootContext)
SP_CLASS_METHOD(BaseContext,getTime)
SP_CLASS_METHOD(BaseContext,getDt)
SP_CLASS_METHOD(BaseContext,getGravity)
SP_CLASS_METHOD(BaseContext,setGravity)
SP_CLASS_METHOD_KW(BaseContext,createObject)
SP_CLASS_METHOD_KW(BaseContext,createObject_noWarning)
SP_CLASS_METHOD(BaseContext,getObject)
SP_CLASS_METHOD(BaseContext,getObjects)
SP_CLASS_METHODS_END


extern "C" PyObject * BaseContext_getAttr_animate(PyObject *self, void*)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return PyBool_FromLong(obj->getAnimate());
}
extern "C" int BaseContext_setAttr_animate(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    if (!PyBool_Check(args))
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->setAnimate(args==Py_True);
    return 0;
}

extern "C" PyObject * BaseContext_getAttr_active(PyObject *self, void*)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    return PyBool_FromLong(obj->isActive());
}
extern "C" int BaseContext_setAttr_active(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj=((PySPtr<Base>*)self)->object->toBaseContext();
    if (!PyBool_Check(args))
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->setActive(args==Py_True);
    return 0;
}

SP_CLASS_ATTRS_BEGIN(BaseContext)
SP_CLASS_ATTR(BaseContext,active)
SP_CLASS_ATTR(BaseContext,animate)
//SP_CLASS_ATTR(BaseContext,gravity) // attribut objets = problème... le setter ne fonctionne pas
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_SPTR_ATTR(BaseContext,BaseContext,Base)
