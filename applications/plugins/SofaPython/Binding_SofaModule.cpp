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
#include "initSofaPython.h"

#include "Binding_SofaModule.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_Node.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/BaseViewer.h>
#include <sofa/gui/GUIManager.h>



using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::component;

#include <sofa/simulation/common/Node.h>
using namespace sofa::simulation;


// set the viewer resolution
extern "C" PyObject * Sofa_getSofaPythonVersion(PyObject * /*self*/, PyObject *)
{
    return Py_BuildValue("s", getModuleVersion());
}


// object factory
extern "C" PyObject * Sofa_createObject(PyObject * /*self*/, PyObject * args, PyObject * kw)
{
    char *type;
    if (!PyArg_ParseTuple(args, "s",&type))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    SP_MESSAGE_WARNING( "Sofa.createObject is deprecated; use Sofa.Node.createObject instead." )

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
        SP_MESSAGE_ERROR( "createObject "<<desc.getName().c_str()<<" of type "<<desc.getAttribute("type","") )
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
    SP_MESSAGE_ERROR( "Sofa.getObject(BaseContext,path) is deprecated. Please use BaseContext.getObject(path) instead." )
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
        SP_MESSAGE_ERROR( "Sofa.getChildNode("<<path<<") not found." )
        return 0;
    }
    return SP_BUILD_PYSPTR(childNode);
    */
    // deprecated on date 2012/07/18
    SP_MESSAGE_ERROR( "Sofa.getChildNode(Node,path) is deprecated. Please use Node.getChild(path) instead." )
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
        SP_MESSAGE_ERROR( "sendGUIMessage("<<msgType<<","<<msgValue<<"): no GUI!" )
        return Py_BuildValue("i",-1);
    }
    gui->sendMessage(msgType,msgValue);


    Py_RETURN_NONE;
}

// ask the GUI to save a screenshot
extern "C" PyObject * Sofa_saveScreenshot(PyObject * /*self*/, PyObject * args)
{
    char *filename;
    if (!PyArg_ParseTuple(args, "s",&filename))
    {
        PyErr_BadArgument();
        return 0;
    }
    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        SP_MESSAGE_ERROR( "saveScreenshot("<<filename<<"): no GUI!" )
        return Py_BuildValue("i",-1);
    }
    gui->saveScreenshot(filename);


    return Py_BuildValue("i",0);
}


// set the viewer resolution
extern "C" PyObject * Sofa_setViewerResolution(PyObject * /*self*/, PyObject * args)
{
	int width, height;
    if (!PyArg_ParseTuple(args, "ii",&width,&height))
    {
        PyErr_BadArgument();
        return 0;
    }
    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        SP_MESSAGE_ERROR( "setViewerResolution("<<width<<","<<height<<"): no GUI!" )
        return Py_BuildValue("i",-1);
    }
    gui->setViewerResolution(width,height);


    return Py_BuildValue("i",0);
}


// set the viewer resolution
extern "C" PyObject * Sofa_setViewerBackgroundColor(PyObject * /*self*/, PyObject * args)
{
	float r = 0.0f, g = 0.0f, b = 0.0f;
	sofa::defaulttype::Vector3 color;
    if (!PyArg_ParseTuple(args, "fff", &r, &g, &b))
    {
        PyErr_BadArgument();
        return 0;
    }
	color[0] = r; color[1] = g; color[2] = b;
	for (int i = 0; i < 3; ++i){
		if (color[i] < 00.f || color[i] > 1.0) {
			PyErr_BadArgument();
			return 0;
		}
	}

    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        SP_MESSAGE_ERROR( "setViewerBackgroundColor("<<r<<","<<g<<","<<b<<"): no GUI!" )
        return Py_BuildValue("i",-1);
    }
    gui->setBackgroundColor(color);


    return Py_BuildValue("i",0);
}

// set the viewer camera
extern "C" PyObject * Sofa_setViewerCamera(PyObject * /*self*/, PyObject * args)
{
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 1.0f;

    if (!PyArg_ParseTuple(args, "fffffff", &px, &py, &pz, &qx, &qy, &qz, &qw))
    {
        PyErr_BadArgument();
        return 0;
    }


    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        SP_MESSAGE_ERROR( "setViewerCamera: no GUI!" )
        return Py_BuildValue("i",-1);
    }
    BaseViewer * viewer = gui->getViewer();
    if (!viewer)
    {
        SP_MESSAGE_ERROR( "setViewerCamera: no Viewer!" )
        return Py_BuildValue("i",-1);
    }
    viewer->setView(sofa::defaulttype::Vector3(px,py,pz),sofa::defaulttype::Quat(qx,qy,qz,qw));

    return Py_BuildValue("i",0);
}


extern "C" PyObject * Sofa_getViewerCamera(PyObject * /*self*/, PyObject *)
{
    sofa::defaulttype::Vector3 pos;
    sofa::defaulttype::Quat orient;


    BaseGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        SP_MESSAGE_ERROR( "getViewerCamera: no GUI!" )
        return Py_BuildValue("i",-1);
    }
    BaseViewer * viewer = gui->getViewer();
    if (!viewer)
    {
        SP_MESSAGE_ERROR( "getViewerCamera: no Viewer! ")
        return Py_BuildValue("i",-1);
    }
    viewer->getView(pos,orient);

    return Py_BuildValue("fffffff",pos.x(),pos.y(),pos.z(),orient[0],orient[1],orient[2],orient[3]);
}



// Méthodes du module
SP_MODULE_METHODS_BEGIN(Sofa)
SP_MODULE_METHOD(Sofa,getSofaPythonVersion) 
SP_MODULE_METHOD_KW(Sofa,createObject)
SP_MODULE_METHOD(Sofa,getObject)        // deprecated on date 2012/07/18
SP_MODULE_METHOD(Sofa,getChildNode)     // deprecated on date 2012/07/18
SP_MODULE_METHOD(Sofa,sendGUIMessage)
SP_MODULE_METHOD(Sofa,saveScreenshot)
SP_MODULE_METHOD(Sofa,setViewerResolution)
SP_MODULE_METHOD(Sofa,setViewerBackgroundColor)
SP_MODULE_METHOD(Sofa,setViewerCamera)
SP_MODULE_METHOD(Sofa,getViewerCamera)
SP_MODULE_METHODS_END



