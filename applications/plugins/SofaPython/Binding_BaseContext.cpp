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


#include "Binding_BaseContext.h"
#include "Binding_Base.h"
#include "Binding_Vector.h"
#include "PythonFactory.h"

#include <sofa/defaulttype/Vec3Types.h>
using namespace sofa::defaulttype;
#include <sofa/core/ObjectFactory.h>
using namespace sofa::core;
#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;
#include <sofa/simulation/Node.h>
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
    return sofa::PythonFactory::toPython(obj->getRootContext());
}

// object factory
extern "C" PyObject * BaseContext_createObject_Impl(PyObject * self, PyObject * args, PyObject * kw, bool printWarnings)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();

    char *type;
    if (!PyArg_ParseTuple(args, "s",&type))
    {
        PyErr_BadArgument();
        return NULL;
    }

    // temporarily, the name is set to the type name.
    // if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type,type);

    bool warning = printWarnings;
    if (kw && PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            PyObject *key = PyList_GetItem(keys,i);
            PyObject *value = PyList_GetItem(values,i);

            if( !strcmp( PyString_AsString(key), "warning") )
            {
                if PyBool_Check(value)
                    warning = (value==Py_True);
            }
            else
            {
                if (PyString_Check(value))
                    desc.setAttribute(PyString_AsString(key),PyString_AsString(value));
                else
                    desc.setAttribute(PyString_AsString(key),PyString_AsString(PyObject_Str(value)));
            }
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,&desc);
    if (obj==0)
    {
        SP_MESSAGE_ERROR( "createObject: component '" << desc.getName() << "' of type '" << desc.getAttribute("type","")<< "' in node '"<<context->getName()<<"'" );
        for (std::vector< std::string >::const_iterator it = desc.getErrors().begin(); it != desc.getErrors().end(); ++it)
            SP_MESSAGE_ERROR(*it);
        PyErr_BadArgument();
        return NULL;
    }


    if( warning )
    {
        for( auto it : desc.getAttributeMap() )
        {
            if (!it.second.isAccessed())
            {
                obj->serr <<"Unused Attribute: \""<<it.first <<"\" with value: \"" <<(std::string)it.second<<"\"" << obj->sendl;
            }
        }

        Node *node = static_cast<Node*>(context);
        if (node && node->isInitialized())
            SP_MESSAGE_WARNING( "Sofa.Node.createObject("<<type<<") called on a node("<<node->getName()<<") that is already initialized" )
    }

    return sofa::PythonFactory::toPython(obj.get());
}
extern "C" PyObject * BaseContext_createObject(PyObject * self, PyObject * args, PyObject * kw)
{
    return BaseContext_createObject_Impl( self, args, kw, true );
}
extern "C" PyObject * BaseContext_createObject_noWarning(PyObject * self, PyObject * args, PyObject * kw)
{
    SP_MESSAGE_DEPRECATED("BaseContext_createObject_noWarning is deprecated, use the keyword warning=False in BaseContext_createObject instead.")
    return BaseContext_createObject_Impl( self, args, kw, false );
}

/// the complete relative path to the object must be given
/// returns None with a warning if the object is not found
extern "C" PyObject * BaseContext_getObject(PyObject * self, PyObject * args, PyObject * kw)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
    {
        SP_MESSAGE_WARNING( "BaseContext_getObject: wrong argument, should be a string (the complete relative path)" )
        Py_RETURN_NONE;
    }

    bool warning = true;
    if (kw && PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            PyObject *key = PyList_GetItem(keys,i);
            PyObject *value = PyList_GetItem(values,i);
            if( !strcmp(PyString_AsString(key),"warning") )
            {
                if PyBool_Check(value)
                    warning = (value==Py_True);
                break;
            }
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    if (!context || !path)
    {
        PyErr_BadArgument();
        return NULL;
    }
    BaseObject::SPtr sptr;
    context->get<BaseObject>(sptr,path);
    if (!sptr)
    {
        if(warning) SP_MESSAGE_WARNING( "BaseContext_getObject: component "<<path<<" not found (the complete relative path is needed)" )
        Py_RETURN_NONE;
    }

    return sofa::PythonFactory::toPython(sptr.get());
}


/// the complete relative path to the object must be given
/// returns None if the object is not found
extern "C" PyObject * BaseContext_getObject_noWarning(PyObject * self, PyObject * args)
{
    SP_MESSAGE_DEPRECATED("BaseContext_getObject_noWarning is deprecated, use the keyword warning=False in BaseContext_getObject instead.")
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
    {
        SP_MESSAGE_WARNING( "BaseContext_getObject_noWarning: wrong argument, should be a string (the complete relative path)" )
        Py_RETURN_NONE;
    }
    if (!context || !path)
    {
        PyErr_BadArgument();
        return NULL;
    }
    BaseObject::SPtr sptr;
    context->get<BaseObject>(sptr,path);
    if (!sptr) Py_RETURN_NONE;

    return sofa::PythonFactory::toPython(sptr.get());
}




// @TODO: pass keyword arguments rather than optional arguments?
extern "C" PyObject * BaseContext_getObjects(PyObject * self, PyObject * args)
{
    BaseContext* context=((PySPtr<Base>*)self)->object->toBaseContext();
    char* search_direction= NULL;
    char* type_name= NULL;
    char* name= NULL;
    if ( !PyArg_ParseTuple ( args, "|sss", &search_direction, &type_name, &name ) ) {
        SP_MESSAGE_WARNING( "BaseContext_getObjects: wrong arguments! Expected format: getObjects ( OPTIONAL STRING searchDirection, OPTIONAL STRING typeName, OPTIONAL STRING name )" )
        Py_RETURN_NONE;
    }

    if (!context)
    {
        PyErr_BadArgument();
        return NULL;
    }

    sofa::core::objectmodel::BaseContext::SearchDirection search_direction_enum= sofa::core::objectmodel::BaseContext::Local;
    if ( search_direction )
    {
        std::string search_direction_str ( search_direction );
        if ( search_direction_str == "SearchUp" )
        {
            search_direction_enum= sofa::core::objectmodel::BaseContext::SearchUp;
        }
        else if ( search_direction_str == "Local" )
        {
            search_direction_enum= sofa::core::objectmodel::BaseContext::Local;
        }
        else if ( search_direction_str == "SearchDown" )
        {
            search_direction_enum= sofa::core::objectmodel::BaseContext::SearchDown;
        }
        else if ( search_direction_str == "SearchRoot" )
        {
            search_direction_enum= sofa::core::objectmodel::BaseContext::SearchRoot;
        }
        else if ( search_direction_str == "SearchParents" )
        {
            search_direction_enum= sofa::core::objectmodel::BaseContext::SearchParents;
        }
        else
        {
            SP_MESSAGE_WARNING( "BaseContext_getObjects: Invalid search direction, using 'Local'. Expected: 'SearchUp', 'Local', 'SearchDown', 'SearchRoot', or 'SearchParents'." )
        }
    }

    sofa::helper::vector< boost::intrusive_ptr<BaseObject> > list;
    context->get<BaseObject>(&list,search_direction_enum);

    PyObject *pyList = PyList_New(0);
    for (size_t i=0; i<list.size(); i++)
    {
        BaseObject* o = list[i].get();

        if( !type_name || o->getClass()->hasParent( type_name ) )
        {
            if ( !name || name == o->getName() )
            {
                PyObject* obj=sofa::PythonFactory::toPython(o); // ref 1
                PyList_Append(pyList,obj); // ref 2
                Py_DECREF(obj); // ref 1 (now owned by list)
            }
        }
    }

    return pyList;
}

SP_CLASS_METHODS_BEGIN(BaseContext)
SP_CLASS_METHOD(BaseContext,getRootContext)
SP_CLASS_METHOD(BaseContext,getTime)
SP_CLASS_METHOD(BaseContext,getDt)
SP_CLASS_METHOD(BaseContext,getGravity)
SP_CLASS_METHOD(BaseContext,setGravity)
SP_CLASS_METHOD_KW(BaseContext,createObject)
SP_CLASS_METHOD_KW(BaseContext,createObject_noWarning) // deprecated
SP_CLASS_METHOD_KW(BaseContext,getObject)
SP_CLASS_METHOD(BaseContext,getObject_noWarning) // deprecated
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
