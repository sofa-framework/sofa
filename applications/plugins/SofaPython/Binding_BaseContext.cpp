/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "PythonToSofa.inl"

#include <sofa/defaulttype/Vec3Types.h>
using namespace sofa::defaulttype;

#include <sofa/core/ObjectFactory.h>
using namespace sofa::core;

#include <sofa/core/objectmodel/BaseContext.h>
using namespace sofa::core::objectmodel;

#include <sofa/simulation/Node.h>
using namespace sofa::simulation;
using namespace sofa::defaulttype;

static inline BaseContext* get_basecontext(PyObject* obj) {
    return sofa::py::unwrap<BaseContext>(obj);
}


static PyObject * BaseContext_setGravity(PyObject *self, PyObject * args)
{
    BaseContext* obj = get_basecontext( self );
    PyPtr<Vector3>* pyVec;
    if (!PyArg_ParseTuple(args, "O",&pyVec)) {
        return NULL;
    }

    obj->setGravity(*pyVec->object);
    Py_RETURN_NONE;
}

static PyObject * BaseContext_getGravity(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj = get_basecontext( self );
    return SP_BUILD_PYPTR(Vector3,Vector3,new Vector3(obj->getGravity()),true); // "true", because I manage the deletion myself
}

static PyObject * BaseContext_getTime(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj = get_basecontext( self );
    return PyFloat_FromDouble(obj->getTime());
}

static PyObject * BaseContext_getDt(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj = get_basecontext( self );
    return PyFloat_FromDouble(obj->getDt());
}

static PyObject * BaseContext_getRootContext(PyObject *self, PyObject * /*args*/)
{
    BaseContext* obj = get_basecontext( self );
    return sofa::PythonFactory::toPython(obj->getRootContext());
}



/// object factory
static PyObject * BaseContext_createObject_Impl(PyObject * self, PyObject * args, PyObject * kw, bool printWarnings)
{
    BaseContext* context = get_basecontext( self );

    char *type;
    if (!PyArg_ParseTuple(args, "s",&type))
    {
        return NULL;
    }

    /// temporarily, the name is set to the type name.
    /// if a "name" parameter is provided, it will overwrite it.
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
                std::stringstream s;
                pythonToSofaDataString(value, s) ;
                desc.setAttribute(PyString_AsString(key),s.str().c_str());
            }
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,&desc);
    if (obj==0)
    {
        std::stringstream msg;
        msg << "Unable to create '" << desc.getName() << "' of type '" << desc.getAttribute("type","")<< "' in node '"<<context->getName()<<"'." ;
        for (std::vector< std::string >::const_iterator it = desc.getErrors().begin(); it != desc.getErrors().end(); ++it)
            msg << " " << *it << msgendl ;

        //todo(STC4) do it or remove it ?
        //todo(dmarchal 2017/10/01) I don't like that because it is weird to have error reporting
        //strategy into the createObject implementation instead of into a dedicated exception.
        //in addition this is the first time we are not using the macro from msg_*
        BaseObjectDescription desc("InfoComponent", "InfoComponent") ;
        BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,&desc) ;
        obj->setName( "Not created ("+std::string(type)+")" ) ;
        sofa::helper::logging::Message m(sofa::helper::logging::Message::Runtime,
                                         sofa::helper::logging::Message::Error) ;
        m << msg.str() ;
        obj->addMessage(  m ) ;
        //todo(STC4) end of do it or remove it.

        PyErr_SetString(PyExc_RuntimeError, msg.str().c_str()) ;
        return NULL;
    }

    if( warning )
    {
        for( auto it : desc.getAttributeMap() )
        {
            if (!it.second.isAccessed())
            {
                obj->serr <<"Unused Attribute: \""<<it.first <<"\" with value: \"" <<(std::string)it.second<<"\" (" << obj->getPathName() << ")" << obj->sendl;
            }
        }

        Node *node = static_cast<Node*>(context);
        if (node && node->isInitialized())
            msg_warning(node) << "Sofa.Node.createObject("<<type<<") called on a node("<<node->getName()<<") that is already initialized";
    }

    return sofa::PythonFactory::toPython(obj.get());
}

static PyObject * BaseContext_createObject(PyObject * self, PyObject * args, PyObject * kw)
{
    return BaseContext_createObject_Impl( self, args, kw, true );
}

static PyObject * BaseContext_createObject_noWarning(PyObject * self, PyObject * args, PyObject * kw)
{
    BaseContext* context = get_basecontext( self );
    msg_deprecated(context)
            << "BaseContext_createObject_noWarning is deprecated, use the keyword warning=False in BaseContext_createObject instead." ;

    return BaseContext_createObject_Impl( self, args, kw, false );
}

/// the complete relative path to the object must be given
/// returns None with a warning if the object is not found
static PyObject * BaseContext_getObject(PyObject * self, PyObject * args, PyObject * kw)
{
    BaseContext* context = get_basecontext( self );
    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
    {
        return NULL;
    }

    bool emitWarningMessage = true;

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
                        emitWarningMessage = (value==Py_True);
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
        return NULL;
    }

    return sofa::PythonFactory::toPython(sptr.get());
}


/// the complete relative path to the object must be given
/// returns None if the object is not found
static PyObject * BaseContext_getObject_noWarning(PyObject * self, PyObject * args)
{
    BaseContext* context = get_basecontext( self );
    msg_deprecated(context)
            << "BaseContext_getObject_noWarning is deprecated, use the keyword warning=False in BaseContext_getObject instead." ;

    char *path;
    if (!PyArg_ParseTuple(args, "s",&path))
    {
        return NULL;
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

//TODO(PR:304) do it or remove  :)
// @TODO: pass keyword arguments rather than optional arguments?
static PyObject * BaseContext_getObjects(PyObject * self, PyObject * args)
{
    BaseContext* context = get_basecontext( self );
    char* search_direction= NULL;
    char* type_name= NULL;
    char* name= NULL;
    if ( !PyArg_ParseTuple ( args, "|sss", &search_direction, &type_name, &name ) ) {
        return NULL;
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
            msg_warning(context) << "BaseContext_getObjects: Invalid search direction, using 'Local'. Expected: 'SearchUp', 'Local', 'SearchDown', 'SearchRoot', or 'SearchParents'." ;
        }
    }

    sofa::helper::vector< BaseObject::SPtr > list;
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
SP_CLASS_METHOD_DOC(BaseContext,getRootContext,
                "Returns the root context of the Sofa Scene.\n"
                "example:\n"
                "   root = node.getRootContext()\n"
                "   root.animate=false")
SP_CLASS_METHOD_DOC(BaseContext,getTime,
                "Returns the accumulated time since the beginnning of the simulation.\n"
                "example:\n"
                "   time = node.getTime()\n"
                "   ")
SP_CLASS_METHOD_DOC(BaseContext,getDt,
                "Returns the current timestep used in the simulation.\n"
                "example:\n"
                "   time = node.getDt()\n"
                "   ")
SP_CLASS_METHOD_DOC(BaseContext,getGravity,
                "Returns the gravity that is applied to node's children (a Sofa.Vector3 object).\n"
                "example:\n"
                "   g = node.getGravity()\n"
                "   print(str(g.x)"
                )
SP_CLASS_METHOD_DOC(BaseContext,setGravity,
                "Sets the gravity applied to the node (a Sofa.Vector3 object).\n"
                "example:\n"
                "   g = Sofa.Vector3(0.0,-9.81,0.0)\n"
                "   node.setGravity(g)"
                )
SP_CLASS_METHOD_KW_DOC(BaseContext,createObject,
               "Creates a Sofa object and then adds it to the node. "
               "First argument is the type name, parameters are passed as subsequent keyword arguments.\n"
               "Automatic conversion is performed for Scalar, Integer, String, List & Sequence as well as \n"
               "object with a getAsCreateObjectParameter(self)."
               "example:\n"
               "   object = node.createObject('MechanicalObject',name='mObject', dx=1, dy=2, dz=3)"
               )
SP_CLASS_METHOD_KW_DOC(BaseContext,createObject_noWarning,   // deprecated
               "(Deprecated) Creates a Sofa object and then adds it to the node. "
               "First argument is the type name, parameters are passed as subsequent keyword arguments. \n"
               "IMPORTANT: In this version, no warning is output in the console if the object cannot be initialized.\n"
               "example:\n"
               "   object = node.createObject_noWarning('MechanicalObject',name='mObject',dx='x',dy='y',dz='z')"
               )
SP_CLASS_METHOD_KW_DOC(BaseContext,getObject,
                "Returns the object by its path. Can be in this node or another, in function of the path... \n"
                "examples:\n"
                "   mecanicalState = node.getObject('DOFs')\n"
                "   mesh = node.getObject('visuNode/OglModel')"
                )
SP_CLASS_METHOD_DOC(BaseContext,getObject_noWarning,
                "(Deprecated) Returns the object by its path. Can be in this node or another, in function of the path... \n"
                "IMPORTANT: In this version, no warning is output in the console if the object cannot be initialized.\n"
                "examples:\n"
                "   mecanicalState = node.getObject_noWarning('DOFs')\n"
                "   mesh = node.getObject('visuNode/OglModel')"

                ) // deprecated
SP_CLASS_METHOD_DOC(BaseContext,getObjects,
                "Returns a list of the objects of this node. \n"
                "example:\n"
                "   objects = node.getObjects()\n"
                "   for obj in objets:\n"
                "       print (obj.name)"
                )
SP_CLASS_METHODS_END


static PyObject * BaseContext_getAttr_animate(PyObject *self, void*)
{
    BaseContext* obj = get_basecontext( self );
    return PyBool_FromLong(obj->getAnimate());
}
static int BaseContext_setAttr_animate(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj = get_basecontext( self );
    if (!PyBool_Check(args))
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->setAnimate(args==Py_True);
    return 0;
}

static PyObject * BaseContext_getAttr_active(PyObject *self, void*)
{
    BaseContext* obj = get_basecontext( self );
    return PyBool_FromLong(obj->isActive());
}

static int BaseContext_setAttr_active(PyObject *self, PyObject * args, void*)
{
    BaseContext* obj = get_basecontext( self );
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
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_SPTR_ATTR(BaseContext,BaseContext,Base)
