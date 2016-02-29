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
#include "PythonMacros.h"
#include <SofaPython/config.h>

#include "Binding_SofaModule.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_Node.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/BaseViewer.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/GenerateRigid.h>
#include <sofa/simulation/common/Simulation.h>
//#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include "ScriptEnvironment.h"
#include <sofa/helper/logging/Messaging.h>


using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::component;

#include <sofa/simulation/common/Node.h>
using namespace sofa::simulation;


// set the viewer resolution
extern "C" PyObject * Sofa_getSofaPythonVersion(PyObject * /*self*/, PyObject *)
{
    return Py_BuildValue("s", SOFAPYTHON_VERSION_STR);
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

    SP_MESSAGE_DEPRECATED( "Sofa.createObject is deprecated; use Sofa.Node.createObject instead." )

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

    // by default, it will always be at least a BaseObject...
    return SP_BUILD_PYSPTR(obj.get());
}


extern "C" PyObject * Sofa_getObject(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getObject(BaseContext,path) is deprecated. Please use BaseContext.getObject(path) instead." )
    PyErr_BadArgument();
    Py_RETURN_NONE;

}

extern "C" PyObject * Sofa_getChildNode(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getChildNode(Node,path) is deprecated. Please use Node.getChild(path) instead." )
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



// from a mesh, a density and a 3d scale
// computes a mass, a center of mass, a diagonal inertia matrix and an inertia rotation
extern "C" PyObject * Sofa_generateRigid(PyObject * /*self*/, PyObject * args)
{
    char* meshFilename;
    double density;
    double sx,sy,sz;
    double rx,ry,rz;
    if (!PyArg_ParseTuple(args, "sddddddd",&meshFilename,&density,&sx,&sy,&sz,&rx,&ry,&rz))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    sofa::helper::GenerateRigidInfo rigid;
    if( !sofa::helper::generateRigid( rigid, meshFilename, density, Vector3(sx,sy,sz), Vector3(rx,ry,rz) ) )
        exit(0);

    return Py_BuildValue("ddddddddddd",rigid.mass
                         ,rigid.com[0],rigid.com[1],rigid.com[2]
                         ,rigid.inertia_diagonal[0],rigid.inertia_diagonal[1],rigid.inertia_diagonal[2]
                         ,rigid.inertia_rotation[0],rigid.inertia_rotation[1],rigid.inertia_rotation[2],rigid.inertia_rotation[3]
            );
}


/// save a sofa scene from python
extern "C" PyObject * Sofa_exportGraph(PyObject * /*self*/, PyObject * args)
{
    char* filename;
    if (!PyArg_ParseTuple(args, "s",&filename))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    getSimulation()->exportGraph( Simulation::GetRoot().get(), filename );

    return Py_BuildValue("i",0);
}



extern "C" PyObject * Sofa_updateVisual(PyObject * /*self*/, PyObject * /*args*/)
{
    Simulation* simulation = getSimulation();
    Node*root=simulation->GetRoot().get();

    //    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
//    root->execute<UpdateBoundingBoxVisitor>(params);
//    simulation->updateVisualContext(root);
    simulation->updateVisual(root);


    Py_RETURN_NONE;
}

// sometimes nodes created in Python are not initialized ASAP
// e.g. when created in callbacks initGraph or bwdInitGraph
// this function forces the initialization of every nodes created in python that are not yet initialized
extern "C" PyObject * Sofa_forceInitNodeCreatedInPython(PyObject * /*self*/, PyObject * /*args*/)
{
    sofa::simulation::ScriptEnvironment::initScriptNodes();
    Py_RETURN_NONE;
}


static const std::string s_emitter = "PythonScript";

extern "C" PyObject * Sofa_msg_info(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    char* message;

    if( argSize==2 )
    {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_info( emitter ) << message;
    }
    else // no emitter
    {
        if( !PyArg_ParseTuple(args, "s", &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_info( s_emitter ) << message;
    }

    Py_RETURN_NONE;
}

extern "C" PyObject * Sofa_msg_deprecated(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    char* message;

    if( argSize==2 )
    {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_deprecated( emitter ) << message;
    }
    else // no emitter
    {
        if( !PyArg_ParseTuple(args, "s", &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_deprecated( s_emitter ) << message;
    }

    Py_RETURN_NONE;
}

extern "C" PyObject * Sofa_msg_warning(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    char* message;

    if( argSize==2 )
    {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_warning( emitter ) << message;
    }
    else // no emitter
    {
        if( !PyArg_ParseTuple(args, "s", &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_warning( s_emitter ) << message;
    }

    Py_RETURN_NONE;
}

extern "C" PyObject * Sofa_msg_error(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    char* message;

    if( argSize==2 )
    {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_error( emitter ) << message;
    }
    else // no emitter
    {
        if( !PyArg_ParseTuple(args, "s", &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_error( s_emitter ) << message;
    }

    Py_RETURN_NONE;
}

extern "C" PyObject * Sofa_msg_fatal(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    char* message;

    if( argSize==2 )
    {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_fatal( emitter ) << message;
    }
    else // no emitter
    {
        if( !PyArg_ParseTuple(args, "s", &message) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        msg_fatal( s_emitter ) << message;
    }

    Py_RETURN_NONE;
}


// Methods of the module
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
SP_MODULE_METHOD(Sofa,generateRigid)
SP_MODULE_METHOD(Sofa,exportGraph)
SP_MODULE_METHOD(Sofa,updateVisual)
SP_MODULE_METHOD(Sofa,forceInitNodeCreatedInPython)
SP_MODULE_METHOD(Sofa,msg_info)
SP_MODULE_METHOD(Sofa,msg_deprecated)
SP_MODULE_METHOD(Sofa,msg_warning)
SP_MODULE_METHOD(Sofa,msg_error)
SP_MODULE_METHOD(Sofa,msg_fatal)
SP_MODULE_METHODS_END



