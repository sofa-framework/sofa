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
#include "PythonMacros.h"
#include <SofaPython/config.h>

#include "Binding_SofaModule.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_Node.h"
#include "PythonFactory.h"
#include "PythonToSofa.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/BaseViewer.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/GenerateRigid.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/SceneLoaderFactory.h>

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::ComponentInfo ;
using sofa::helper::logging::SofaComponentInfo ;
#include <sofa/helper/Utils.h>
#include "SceneLoaderPY.h"

#include <sofa/helper/system/PluginManager.h>

using namespace sofa::helper;

using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::component;

#include <sofa/simulation/Node.h>
using namespace sofa::simulation;

using sofa::helper::Utils ;
using sofa::helper::system::PluginManager ;

using sofa::simulation::Node ;
using sofa::simulation::SceneLoaderPY ;

using sofa::gui::BaseGUI ;
using sofa::gui::GUIManager ;
using sofa::gui::BaseViewer ;

using sofa::PythonFactory ;



/// set the viewer resolution
static PyObject * Sofa_getSofaPythonVersion(PyObject * /*self*/, PyObject *)
{
    return Py_BuildValue("s", SOFAPYTHON_VERSION_STR);
}

static PyObject * Sofa_createNode(PyObject * /*self*/, PyObject * args)
{
    char *name;
    if (!PyArg_ParseTuple(args, "s",&name)) {
        return NULL;
    }

    Node::SPtr node = Node::create( name );

    return PythonFactory::toPython(node.get());
}


/// object factory
static PyObject * Sofa_createObject(PyObject * /*self*/, PyObject * args, PyObject * kw) {
    char *type;
    if (!PyArg_ParseTuple(args, "s", &type)) {
        return NULL;
    }

    SP_MESSAGE_DEPRECATED( "Sofa.createObject is deprecated; use Sofa.Node.createObject instead." )

    /// temporarily, the name is set to the type name.
    /// if a "name" parameter is provided, it will overwrite it.
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

    /// by default, it will always be at least a BaseObject...
    return PythonFactory::toPython(obj.get());
}


static PyObject * Sofa_getObject(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getObject(BaseContext,path) is deprecated. Please use BaseContext.getObject(path) instead." )
    PyErr_BadArgument();
    return NULL;

}

static PyObject * Sofa_getChildNode(PyObject * /*self*/, PyObject * /*args*/)
{
    // deprecated on date 2012/07/18
    SP_MESSAGE_DEPRECATED( "Sofa.getChildNode(Node,path) is deprecated. Please use Node.getChild(path) instead." )
    PyErr_BadArgument();
    return NULL;
}


/// send a text message to the GUI
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


    return NULL;
}

/// ask the GUI to save a screenshot
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


/// set the viewer resolution
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


/// set the viewer resolution
static PyObject * Sofa_setViewerBackgroundColor(PyObject * /*self*/, PyObject * args)
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

/// set the viewer camera
static PyObject * Sofa_setViewerCamera(PyObject * /*self*/, PyObject * args)
{
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 1.0f;

	if (!PyArg_ParseTuple(args, "fffffff", &px, &py, &pz, &qx, &qy, &qz, &qw)) {
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



/// from a mesh, a density and a 3d scale, computes a mass, a center of mass, a
/// diagonal inertia matrix and an inertia rotation
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

    BaseNode* node = sofa::py::unwrap<BaseNode>( pyNode );
    if (!node) {
        SP_PYERR_SETSTRING_INVALIDTYPE("BaseNode*") ;
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

    BaseNode* basenode = sofa::py::unwrap<BaseNode>( pyNode );
    if (!basenode) {
        SP_PYERR_SETSTRING_INVALIDTYPE("BaseNode*") ;
        return NULL;
    }

    Node* node = down_cast<Node>(basenode);
    Simulation* simulation = getSimulation();

    simulation->updateVisual(node);
    return NULL;
}

static const std::string s_emitter = "PythonScript";

template<class Action>
static PyObject* parse_emitter_message_then(PyObject* args, const Action& action) {
    PyObject* py_emitter {nullptr};
    PyObject* py_message {nullptr};

    const size_t argSize = PyTuple_Size(args);

    if( argSize == 1 ) {
        /// no emitter
        char* message;
        if( !PyArg_ParseTuple(args, "s", &message) ) {
            return NULL;
        }

        action(ComponentInfo::SPtr(new ComponentInfo(s_emitter)), message, PythonEnvironment::getPythonCallingPointAsFileInfo());
    } else if( argSize == 2 ) {
        /// SOURCE, "Message"
        if( !PyArg_ParseTuple(args, "OO", &py_emitter, &py_message) ){
            return NULL;
        }
        if( !PyString_Check(py_message) ){
            PyErr_SetString(PyExc_TypeError, "The second parameter must be a string");
            return NULL;
        }

        if( PyString_Check(py_emitter) ){
            action(ComponentInfo::SPtr(new ComponentInfo(PyString_AsString(py_emitter))), PyString_AsString(py_message), PythonEnvironment::getPythonCallingPointAsFileInfo() );
        }else if (PyObject_IsInstance(py_emitter, reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(Base)))) {
            Base* base=(((PySPtr<Base>*)py_emitter)->object).get();
            action(ComponentInfo::SPtr(new SofaComponentInfo(base)), PyString_AsString(py_message), PythonEnvironment::getPythonCallingPointAsFileInfo() );
        }else{
            PyErr_SetString(PyExc_TypeError, "The first parameter must be a string or a Sofa.Base");
            return NULL;
        }
    } else if( argSize == 3 ){
        /// "Message", "FILENAME", LINENO
        char* message;
        char* filename;
        int   lineno;
        if( !PyArg_ParseTuple(args, "ssi", &message, &filename, &lineno) ) {
            return NULL;
        }
        action(ComponentInfo::SPtr(new ComponentInfo(s_emitter)), message, SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
    } else if (argSize == 4 ){
        /// SOURCE, "Message", "FILENAME", LINENO
        char* filename;
        int   lineno;
        if( !PyArg_ParseTuple(args, "OOsi", &py_emitter, &py_message, &filename, &lineno) ){
            return NULL;
        }
        if( !PyString_Check(py_message) ){
            PyErr_SetString(PyExc_TypeError, "The second parameter must be a string");
            return NULL;
        }
        if( PyString_Check(py_emitter) ){
            action(ComponentInfo::SPtr(new ComponentInfo(PyString_AsString(py_emitter))),
                   PyString_AsString(py_message),  SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
        }else if (PyObject_IsInstance(py_emitter, reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(Base)))) {
            Base* base=(((PySPtr<Base>*)py_emitter)->object).get();
            action(ComponentInfo::SPtr(new SofaComponentInfo(base)),
                   PyString_AsString(py_message),  SOFA_FILE_INFO_COPIED_FROM(filename, lineno));
        }else{
            PyErr_SetString(PyExc_TypeError, "The first parameter must be a string or a Sofa.Base");
            return NULL;
        }
    }

	Py_RETURN_NONE;
}

static PyObject * Sofa_msg_info(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, const char* message, const sofa::helper::logging::FileInfo::SPtr& fileinfo) {
        msg_info(emitter) << message << fileinfo;
    });
}

static PyObject * Sofa_msg_deprecated(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, const char* message, const sofa::helper::logging::FileInfo::SPtr& fileinfo) {
        msg_deprecated(emitter) << message << fileinfo;
    });
}

static PyObject * Sofa_msg_warning(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, const char* message, const sofa::helper::logging::FileInfo::SPtr& fileinfo) {
        msg_warning(emitter) << message << fileinfo;
    });
}

static PyObject * Sofa_msg_error(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, const char* message, const sofa::helper::logging::FileInfo::SPtr& fileinfo) {
        msg_error(emitter) << message << fileinfo;
    });
}

static PyObject * Sofa_msg_fatal(PyObject * /*self*/, PyObject * args) {
    return parse_emitter_message_then(args, [](const ComponentInfo::SPtr& emitter, const char* message, const sofa::helper::logging::FileInfo::SPtr& fileinfo) {
        msg_fatal(emitter) << message << fileinfo;
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
        return NULL;

    sofa::simulation::SceneLoader *loader = SceneLoaderFactory::getInstance()->getEntryFileName(filename);

    if (loader)
    {
        Node::SPtr node = loader->load(filename);
        return PythonFactory::toPython(node.get());
    }

    SP_MESSAGE_ERROR( "Sofa_loadScene: extension ("
                      << sofa::helper::system::SetDirectory::GetExtension(filename)<<") not handled" );

    return NULL;
}


static PyObject * Sofa_unload(PyObject * /*self*/, PyObject * args)
{
    PyObject* pyNode;
    if (!PyArg_ParseTuple(args, "O", &pyNode)) {
        return NULL;
    }

    Node* node = sofa::py::unwrap<Node>(pyNode);
    if (!node) {
        return NULL;
    }

    sofa::simulation::getSimulation()->unload( node );

    return NULL;
}

static PyObject * Sofa_loadPythonSceneWithArguments(PyObject * /*self*/, PyObject * args)
{
    size_t argSize = PyTuple_Size(args);

    if( !argSize ) {
        PyErr_SetString(PyExc_RuntimeError, "Sofa_loadPythonSceneWithArguments: should have at least a filename as arguments") ;
        return nullptr;
    }

    char *filename = PyString_AsString(PyTuple_GetItem(args,0));

    if( sofa::helper::system::SetDirectory::GetFileName(filename).empty() ) {// no filename
        PyErr_SetString(PyExc_RuntimeError, "Empty filename.") ;
        return nullptr;
    }

    std::vector<std::string> arguments;
    for( size_t i=1 ; i<argSize ; i++ ) {
        arguments.push_back( PyString_AsString(PyTuple_GetItem(args,i)) );
    }

    SceneLoaderPY loader;
    Node::SPtr root;
    loader.loadSceneWithArguments(filename, arguments, &root);
    return PythonFactory::toPython(root.get());
}

static PyObject * Sofa_loadPlugin(PyObject * /*self*/, PyObject * args)
{
    char *pluginName;
    if (!PyArg_ParseTuple(args, "s", &pluginName)) {
        return NULL;
    }

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
    } else {
        std::stringstream ss;
        ss << "cannot find plugin '" << pluginName  << "'";
        PyErr_SetString(PyExc_EnvironmentError, ss.str().c_str());
        return NULL;
    }

    return PyString_FromString(path.c_str());
}

static PyObject * Sofa_path(PyObject * /*self*/, PyObject * /*args*/) {
    return PyString_FromString(Utils::getSofaPathPrefix().c_str());
}


static PyObject * Sofa_getAvailableComponents(PyObject * /*self*/, PyObject * args)
{
    if(PyTuple_Size(args))
    {
        PyErr_SetString(PyExc_RuntimeError, "This function expects no arguments.");
        return NULL;
    }

    std::vector<ObjectFactory::ClassEntry::SPtr> entries ;
    ObjectFactory::getInstance()->getAllEntries(entries) ;

    PyObject *pyList = PyList_New(entries.size());
    for (size_t i=0; i<entries.size(); i++){
        PyObject *tuple = PyList_New(2);
        PyList_SetItem(tuple, 0, Py_BuildValue("s", entries[i]->className.c_str()));
        PyList_SetItem(tuple, 1, Py_BuildValue("s", entries[i]->description.c_str()));
        PyList_SetItem(pyList, (Py_ssize_t)i, tuple);
    }

    return pyList;
}

static PyObject * Sofa_getAliasesFor(PyObject * /*self*/, PyObject * args)
{
    char* componentname;
    if (!PyArg_ParseTuple(args, "s", &componentname)) {
        return NULL;
    }

    const ObjectFactory::ClassEntry& entry = ObjectFactory::getInstance()->getEntry(componentname) ;

    PyObject *pyList = PyList_New(entry.aliases.size());
    unsigned int i=0;
    for (auto& alias : entry.aliases){
        PyList_SetItem(pyList, (Py_ssize_t)i, Py_BuildValue("s", alias.c_str()));
        i++;
    }

    return pyList;
}

// -----------------


/**
 * Method : Sofa_clear
 * Desc   : Wrapper for python usage. Clear the timer.
 * Param  : PyObject*, self - Object of the python script
 * Return : NULL
 */
static PyObject * Sofa_timerClear(PyObject* /*self*/, PyObject * /*args*/)
{
    AdvancedTimer::clear();  // Method call
    Py_RETURN_NONE;
}


/**
 * Method : Sofa_isEnabled
 * Desc   : Wrapper for python usage. Return if the timer is enable or not.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerIsEnabled(PyObject* /*self*/, PyObject *args)
{
    char* id;
    bool answer = false;

    if(!PyArg_ParseTuple(args, "s", &id))
    {
        return NULL;
    }

    answer = AdvancedTimer::isEnabled(id);  // Method call

    if(answer)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}


/**
 * Method : Sofa_setEnabled
 * Desc   : Wrapper for python usage. /!\ Need to pass an int in arguments insteed of a bool in the python script.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerSetEnabled(PyObject* /*self*/, PyObject *args)
{
    char* id;
    PyObject* val;

    if(!PyArg_ParseTuple(args, "sO", &id, &val))
    {
        Py_RETURN_NONE;
    }

    AdvancedTimer::setEnabled(id, PyObject_IsTrue(val));  // Method call
    Py_RETURN_NONE;
}


/**
 * Method : Sofa_getInterval
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerGetInterval(PyObject* /*self*/, PyObject *args)
{
    char* id;
    int answer = 0;

    if(!PyArg_ParseTuple(args, "s", &id))
    {
        return NULL;
    }

    answer = AdvancedTimer::getInterval(id);  // Method call

    return PyInt_FromLong(static_cast<long int>(answer));
}


/**
 * Method : Sofa_setInterval
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerSetInterval(PyObject* /*self*/, PyObject *args)
{
    char* id;
    int newValue = 0;

    if(!PyArg_ParseTuple(args, "si", &id, &newValue))
    {
        return NULL;
    }

    AdvancedTimer::setInterval(id, newValue);  // Method call


    Py_RETURN_NONE;
}


/**
 * Method : Sofa_begin
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerBegin(PyObject* /*self*/, PyObject *args)
{
    char* id;

    if(!PyArg_ParseTuple(args, "s", &id))
    {
        return NULL;
    }

    AdvancedTimer::begin(id);  // Method call

    Py_RETURN_NONE;
}


/**
 * Method : Sofa_timerStepBegin
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerStepBegin(PyObject*, PyObject* args)
{
    char* id;

    if(!PyArg_ParseTuple(args, "s", &id))
    {
        return NULL;
    }

    AdvancedTimer::stepBegin(id);  // Method call

    Py_RETURN_NONE;
}


/**
 * Method : Sofa_timerStepEnd
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerStepEnd(PyObject*, PyObject* args)
{
    char* id;

    if(!PyArg_ParseTuple(args, "s", &id))
    {
        return NULL;
    }

    AdvancedTimer::stepEnd(id);  // Method call

    Py_RETURN_NONE;
}


/**
 * Method : Sofa_timerEnd
 * Desc   : Wrapper for python usage.
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : string
 */
static PyObject * Sofa_timerEnd(PyObject* /*self*/, PyObject *args)
{
    char* id = NULL;
    void* tempNode = NULL;
    Node* node = NULL;
    std::string result;

    if(!PyArg_ParseTuple(args, "sO", &id, &tempNode))
    {
        return NULL;
    }

    node = down_cast<Node>(((PySPtr<Base>*)tempNode)->object->toBaseNode());

    result = AdvancedTimer::end(id, node);

    if(std::string("null").compare(result) == 0)
        Py_RETURN_NONE;

    return PyString_FromString(result.c_str());  // Method call
}


/**
 * Method : Sofa_timerSetOutputType
 * Desc   : Wrapper for python usage. Used to change output type of the given timer
 * Param  : PyObject*, self - Object of the python script
 * Param  : PyObject*, args - given arguments to apply to the method
 * Return : NULL
 */
static PyObject * Sofa_timerSetOutputType(PyObject* /*self*/, PyObject *args)
{
    char* id = NULL;
    char* newOutputType = NULL;

    if(!PyArg_ParseTuple(args, "ss", &id, &newOutputType))
    {
        return NULL;
    }

    AdvancedTimer::setOutputType(id, newOutputType);

    Py_RETURN_NONE;
}



/// Methods of the module
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
SP_MODULE_METHOD(Sofa,unload)
SP_MODULE_METHOD(Sofa,loadPythonSceneWithArguments)
SP_MODULE_METHOD(Sofa,loadPlugin)
SP_MODULE_METHOD(Sofa,path)
SP_MODULE_METHOD_DOC(Sofa,getAvailableComponents, "Returns the list of the available components in the factory.")
SP_MODULE_METHOD_DOC(Sofa,getAliasesFor, "Returns the list of the aliases for a given component")
SP_MODULE_METHOD_DOC(Sofa, timerClear, "Method : Sofa_clear \nDesc   : Wrapper for python usage. Clear the timer. \nParam  : PyObject*, self - Object of the python script \nReturn : return None")
SP_MODULE_METHOD_DOC(Sofa, timerIsEnabled, "Method : Sofa_isEnabled \nDesc   : Wrapper for python usage. Return if the timer is enable or not. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerSetEnabled, "Method : Sofa_setEnabled \nDesc   : Wrapper for python usage. /!\\ Need to pass an int in arguments insteed of a bool in the python script. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerGetInterval, "Method : Sofa_getInterval \nDesc   : Wrapper for python usage. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerSetInterval, "Method : Sofa_setInterval \nDesc   : Wrapper for python usage. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerBegin, "Method : Sofa_begin \nDesc   : Wrapper for python usage. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : return None")
SP_MODULE_METHOD_DOC(Sofa, timerStepBegin, "Method : Sofa_timerStepBegin \nDesc   : Wrapper for python usage. \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerStepEnd, "Method : Sofa_timerStepEnd \nDesc   : Wrapper for python usage. \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerSetOutputType, "Method : Sofa_timerSetOutputType \nDesc   : Wrapper for python usage. \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : None")
SP_MODULE_METHOD_DOC(Sofa, timerEnd, "Method : Sofa_timerEnd \nDesc   : Wrapper for python usage. Used to change output type of the given timer \nParam  : PyObject*, self - Object of the python script \nParam  : PyObject*, args - given arguments to apply to the method \nReturn : return None")
SP_MODULE_METHODS_END
