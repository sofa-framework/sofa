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
#include "PythonMacros.h"
#include <SofaPython/config.h>

#include "Binding_SofaModule.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_Node.h"
#include "PythonFactory.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/BaseViewer.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/GenerateRigid.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/SceneLoaderFactory.h>
//#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/logging/Messaging.h>

#include "SceneLoaderPY.h"

#include <sofa/helper/system/PluginManager.h>

using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::component;

#include <sofa/simulation/Node.h>
using namespace sofa::simulation;


// set the viewer resolution
extern "C" PyObject * Sofa_getSofaPythonVersion(PyObject * /*self*/, PyObject *)
{
    return Py_BuildValue("s", SOFAPYTHON_VERSION_STR);
}

extern "C" PyObject * Sofa_createNode(PyObject * /*self*/, PyObject * args)
{
    char *name;
    if (!PyArg_ParseTuple(args, "s",&name)) {
        return NULL;
    }

    sofa::simulation::Node::SPtr node = sofa::simulation::Node::create( name );

    return sofa::PythonFactory::toPython(node.get());
}


// object factory
static PyObject * Sofa_createObject(PyObject * /*self*/, PyObject * args, PyObject * kw) {
    char *type;
    if (!PyArg_ParseTuple(args, "s", &type)) {
        return NULL;
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
        return NULL;
    }

    // by default, it will always be at least a BaseObject...
    return sofa::PythonFactory::toPython(obj.get());
}


extern "C" PyObject * Sofa_getObject(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getObject(BaseContext,path) is deprecated. Please use BaseContext.getObject(path) instead." )
    PyErr_BadArgument();
    return NULL;

}

extern "C" PyObject * Sofa_getChildNode(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getChildNode(Node,path) is deprecated. Please use Node.getChild(path) instead." )
    PyErr_BadArgument();
    return NULL;
}

using namespace sofa::gui;

// send a text message to the GUI
static PyObject * Sofa_sendGUIMessage(PyObject * /*self*/, PyObject * args) {
    char *msgType;
    char *msgValue;
    if (!PyArg_ParseTuple(args, "ss",&msgType,&msgValue)) {
        return NULL;
    }
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
static PyObject * Sofa_saveScreenshot(PyObject * /*self*/, PyObject * args) {
    char *filename;
    if (!PyArg_ParseTuple(args, "s",&filename)) {
        return NULL;
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
static PyObject * Sofa_setViewerResolution(PyObject * /*self*/, PyObject * args) {
    int width, height;
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return NULL;
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
    sofa::defaulttype::RGBAColor color;
    if (!PyArg_ParseTuple(args, "fff", &r, &g, &b)) {
        return NULL;
    }

    color[0] = r; color[1] = g; color[2] = b;
    for (int i = 0; i < 3; ++i){
        if (color[i] < 00.f || color[i] > 1.0) {
            PyErr_BadArgument();
            return NULL;
        }
    }

    color.set(r,g,b,1.0);

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
        return NULL;
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


static PyObject * Sofa_getViewerCamera(PyObject * /*self*/, PyObject *) {
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



// from a mesh, a density and a 3d scale, computes a mass, a center of mass, a
// diagonal inertia matrix and an inertia rotation
static PyObject * Sofa_generateRigid(PyObject * /*self*/, PyObject * args) {
    char* meshFilename;
    double density;
    double sx = 1, sy = 1, sz = 1;
    double rx = 0, ry = 0, rz = 0;
    
    if (!PyArg_ParseTuple(args, "sd|dddddd", &meshFilename, &density,
                          &sx, &sy, &sz,
                          &rx, &ry, &rz)) {
        return NULL;
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
static PyObject * Sofa_exportGraph(PyObject * /*self*/, PyObject * args) {
    char* filename;
    PyObject* pyNode;
    if (!PyArg_ParseTuple(args, "Os", &pyNode, &filename)) {
        return NULL;
    }

    BaseNode* node=((PySPtr<Base>*)pyNode)->object->toBaseNode();
    if (!node) {
        // this should not happen
        PyErr_BadArgument();
        return NULL;
    }


    getSimulation()->exportGraph( down_cast<Node>(node), filename );

    return Py_BuildValue("i",0);
}



static PyObject * Sofa_updateVisual(PyObject * /*self*/, PyObject * args) {
    PyObject* pyNode;
    if (!PyArg_ParseTuple(args, "O", &pyNode)) {
        return NULL;
    }

    BaseNode* basenode=((PySPtr<Base>*)pyNode)->object->toBaseNode();
    if (!basenode) {
        // this should not happen
        PyErr_BadArgument();
        return NULL;
    }

    Node* node = down_cast<Node>(basenode);
    Simulation* simulation = getSimulation();

    //    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
//    node->execute<UpdateBoundingBoxVisitor>(params);
//    simulation->updateVisualContext(node);
    simulation->updateVisual(node);


    Py_RETURN_NONE;
}


static const std::string s_emitter = "PythonScript";

// please use functions instead of copypasting all the time god dammit
template<class Action>
static PyObject* parse_emitter_message_then(PyObject* args, const Action& action) {
    const size_t argSize = PyTuple_Size(args);

    char* message;

    // the logic would be to have the optional arg in last position :-/
    if( argSize == 2 ) {
        char* emitter;
        if( !PyArg_ParseTuple(args, "ss", &emitter, &message) ) {
            return NULL;
        }

        action(emitter, message);
    } else { 
        // no emitter
        if( !PyArg_ParseTuple(args, "s", &message) ) {
            return NULL;
        }

        action(s_emitter, message);
    }
    
    Py_RETURN_NONE;
}

// also, we'd probably would be better off having 'error', 'fatal', 'info' as
// argument
static PyObject * Sofa_msg_info(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const std::string& emitter, const char* message) {
            msg_info(emitter) << message;
        });
}

static PyObject * Sofa_msg_deprecated(PyObject * /*self*/, PyObject * args) {

    return parse_emitter_message_then(args, [](const std::string& emitter, const char* message) {
            msg_deprecated(emitter) << message;
        });

}

static PyObject * Sofa_msg_warning(PyObject * /*self*/, PyObject * args) {

    return parse_emitter_message_then(args, [](const std::string& emitter, const char* message) {
            msg_warning(emitter) << message;
        });
    
}

static PyObject * Sofa_msg_error(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const std::string& emitter, const char* message) {
            msg_error(emitter) << message;
        });
    
}

static PyObject * Sofa_msg_fatal(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const std::string& emitter, const char* message) {
            msg_fatal(emitter) << message;
        });
}



static PyObject * Sofa_loadScene(PyObject * /*self*/, PyObject * args)
{
    char *filename;
    if (!PyArg_ParseTuple(args, "s",&filename)) {
        return NULL;
    }

    if( sofa::helper::system::SetDirectory::GetFileName(filename).empty() || // no filename
            sofa::helper::system::SetDirectory::GetExtension(filename).empty() ) // filename with no extension
        Py_RETURN_NONE;

    sofa::simulation::SceneLoader *loader = SceneLoaderFactory::getInstance()->getEntryFileName(filename);

    if (loader)
    {
        sofa::simulation::Node::SPtr node = loader->load(filename);
        return sofa::PythonFactory::toPython(node.get());
    }

    // unable to load file
    SP_MESSAGE_ERROR( "Sofa_loadScene: extension ("
                      << sofa::helper::system::SetDirectory::GetExtension(filename)<<") not handled" );

    Py_RETURN_NONE;
}



extern "C" PyObject * Sofa_loadPythonSceneWithArguments(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    // TODO FIXME this is an error, raise proper exception
    // e.g. PyError_SetString(PyExc_RuntimeError, "derp"); then return NULL;
    if( !argSize ) {
        SP_MESSAGE_ERROR( "Sofa_loadPythonSceneWithArguments: should have at least a filename as arguments" );
        Py_RETURN_NONE;
    }

    // PyString_Check(PyTuple_GetItem(args,0)) // to check the arg type and raise an error
    char *filename = PyString_AsString(PyTuple_GetItem(args,0));

    if( sofa::helper::system::SetDirectory::GetFileName(filename).empty() ) {// no filename
        // TODO FIXME same here
        Py_RETURN_NONE;
    }

    std::vector<std::string> arguments;
    for( size_t i=1 ; i<argSize ; i++ ) {
        arguments.push_back( PyString_AsString(PyTuple_GetItem(args,i)) );
    }

    sofa::simulation::SceneLoaderPY loader;
    sofa::simulation::Node::SPtr node = loader.loadSceneWithArguments(filename,arguments);
    return sofa::PythonFactory::toPython(node.get());
}



extern "C" PyObject * Sofa_loadPlugin(PyObject * /*self*/, PyObject * args)
{
    char *pluginName;
    if (!PyArg_ParseTuple(args, "s", &pluginName)) {
        return NULL;
    }

    using sofa::helper::system::PluginManager;

    PluginManager& pluginManager = PluginManager::getInstance();

    const std::string path = pluginManager.findPlugin(pluginName);
    if (path != "")
    {
        if (!PluginManager::getInstance().pluginIsLoaded(path))
        {
            if (PluginManager::getInstance().loadPlugin(path))
            {
                const std::string guiPath = pluginManager.findPlugin( std::string( pluginName ) + "_" + PluginManager::s_gui_postfix);
                if (guiPath != "")
                {
                    PluginManager::getInstance().loadPlugin(guiPath);
                }
            }
        }
    }
    else
    {
        SP_MESSAGE_WARNING( "Sofa_loadPlugin: cannot find plugin: " << pluginName );
        PyErr_BadArgument();
        return NULL;
    }

    Py_RETURN_NONE;
}




// Methods of the module
SP_MODULE_METHODS_BEGIN(Sofa)
SP_MODULE_METHOD(Sofa,getSofaPythonVersion)
SP_MODULE_METHOD(Sofa,createNode)
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
SP_MODULE_METHOD(Sofa,msg_info)
SP_MODULE_METHOD(Sofa,msg_deprecated)
SP_MODULE_METHOD(Sofa,msg_warning)
SP_MODULE_METHOD(Sofa,msg_error)
SP_MODULE_METHOD(Sofa,msg_fatal)
SP_MODULE_METHOD(Sofa,loadScene)
SP_MODULE_METHOD(Sofa,loadPythonSceneWithArguments)
SP_MODULE_METHOD(Sofa,loadPlugin)
SP_MODULE_METHODS_END



