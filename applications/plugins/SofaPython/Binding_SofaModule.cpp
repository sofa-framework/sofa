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
#include "PythonMacros.h"

#include "Binding_SofaModule.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_Node.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/GUIManager.h>


using namespace sofa::core;
using namespace sofa::core::objectmodel;

#include <sofa/simulation/common/Node.h>
using namespace sofa::simulation;


// object factory
extern "C" PyObject * Sofa_createObject(PyObject * /*self*/, PyObject * args, PyObject * kw)
{
    char *type;
    if (!PyArg_ParseTuple(args, "s",&type))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    printf("<SofaPython> WARNING Sofa.createObject is deprecated; use Sofa.Node.createObject instead.\n");

    // temporarily, the name is set to the type name.
    // if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type,type);

    if (PyDict_Size(kw)>0)
    {
        PyObject* keys = PyDict_Keys(kw);
        PyObject* values = PyDict_Values(kw);
        for (int i=0; i<PyDict_Size(kw); i++)
        {
            desc.setAttribute(PyString_AsString(PyList_GetItem(keys,i)),PyString_AsString(PyList_GetItem(values,i)));
        }
        Py_DecRef(keys);
        Py_DecRef(values);
    }

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(0,&desc);//.get();
    if (obj==0)
    {
        printf("<SofaPython> ERROR createObject '%s' of type '%s''\n",
                desc.getName().c_str(),
                desc.getAttribute("type",""));
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    // par défaut, ce sera toujours au minimum un BaseObject...
    return SP_BUILD_PYSPTR(obj.get());
}


extern "C" PyObject * Sofa_getObject(PyObject * /*self*/, PyObject * /*args*/)
{
    /*
        PyObject* pyContext;
        char *path;
        if (!PyArg_ParseTuple(args, "Os",&pyContext,&path))
            return 0;
        BaseContext *context=dynamic_cast<BaseContext*>(((PySPtr<Base>*)pyContext)->object.get());
        if (!context || !path)
        {
            PyErr_BadArgument();
            return 0;
        }
        BaseObject::SPtr sptr;
        context->get<BaseObject>(sptr,path);

        return SP_BUILD_PYSPTR(sptr.get());
    */
    // deprecated on date 2012/07/18
    printf("<SofaPython> ERROR: Sofa.getObject(BaseContext,path) is deprecated.\nPlease use BaseContext.getObject(path) instead.\n");
    PyErr_BadArgument();
    Py_RETURN_NONE;

}

extern "C" PyObject * Sofa_getChildNode(PyObject * /*self*/, PyObject * /*args*/)
{
    /*
    PyObject* pyBaseNode;
    char *path;
    if (!PyArg_ParseTuple(args, "Os",&pyBaseNode,&path))
        return 0;
    BaseNode *node=dynamic_cast<BaseNode*>(((PySPtr<Base>*)pyBaseNode)->object.get());
    if (!node || !path)
    {
        PyErr_BadArgument();
        return 0;
    }

    const objectmodel::BaseNode::Children& children = node->getChildren();
    Node *childNode = 0;
    // BaseNode ne pouvant pas être bindé en Python, et les BaseNodes des graphes étant toujours des Nodes,
    // on caste directement en Node.
    for (unsigned int i=0; i<children.size(); ++i)
        if (children[i]->getName() == path)
        {
            childNode = dynamic_cast<Node*>(children[i]);
            break;
        }
    if (!childNode)
    {
        printf("<SofaPython> Error: Sofa.getChildNode(%s) not found.\n",path);
        return 0;
    }
    return SP_BUILD_PYSPTR(childNode);
    */
    // deprecated on date 2012/07/18
    printf("<SofaPython> ERROR: Sofa.getChildNode(Node,path) is deprecated.\nPlease use Node.getChild(path) instead.\n");
    PyErr_BadArgument();
    Py_RETURN_NONE;
}

using namespace sofa::gui;

// send a text message to the GUI
extern "C" PyObject * Sofa_sendGUIMessage(PyObject * /*self*/, PyObject * args)
{
    char *msgType;
    char *msgValue;
    if (!PyArg_ParseTuple(args, "ss",&msgType,&msgValue))
        Py_RETURN_NONE;
    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        printf("<SofaPython> ERROR sendGUIMessage(%s,%s): no GUI !!\n",msgType,msgValue);
        return Py_BuildValue("i",-1);
    }
    gui->sendMessage(msgType,msgValue);


    Py_RETURN_NONE;
}


extern "C" PyObject* Sofa_build_dir(PyObject * /*self*/, PyObject * /*args*/ ) {
	return Py_BuildValue("s", sofa_tostring(SOFA_BUILD_DIR));
}

extern "C" PyObject* Sofa_src_dir(PyObject * /*self*/, PyObject * /*args*/ ) {
	return Py_BuildValue("s", sofa_tostring(SOFA_SRC_DIR));
}



// Méthodes du module
SP_MODULE_METHODS_BEGIN(Sofa)
SP_MODULE_METHOD_KW(Sofa,createObject)
SP_MODULE_METHOD(Sofa,getObject)        // deprecated on date 2012/07/18
SP_MODULE_METHOD(Sofa,getChildNode)     // deprecated on date 2012/07/18
SP_MODULE_METHOD(Sofa,sendGUIMessage)
SP_MODULE_METHOD(Sofa,build_dir)
SP_MODULE_METHOD(Sofa,src_dir)
SP_MODULE_METHODS_END



