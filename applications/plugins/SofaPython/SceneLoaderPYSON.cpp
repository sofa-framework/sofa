/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "PythonEnvironment.h"
#include "SceneLoaderPYSON.h"


#include <sofa/simulation/Simulation.h>
#include <SofaSimulationCommon/xml/NodeElement.h>
#include <SofaSimulationCommon/FindByTypeVisitor.h>

#include <sstream>

#include "PythonMainScriptController.h"
#include "PythonEnvironment.h"
#include "PythonFactory.h"

using namespace sofa::core::objectmodel;

namespace sofa
{

namespace simulation
{

namespace _sceneloaderpyson_
{

bool SceneLoaderPYSON::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="pyson");
}

bool SceneLoaderPYSON::canWriteFileExtension(const char *extension)
{
    return canLoadFileExtension(extension);
}

/// get the file type description
std::string SceneLoaderPYSON::getFileTypeDesc()
{
    return "SOFA Json Scenes + Python";
}

/// get the list of file extensions
void SceneLoaderPYSON::getExtensionList(ExtensionList* list)
{
    list->clear();
    list->push_back("pyson");
}

sofa::simulation::Node::SPtr SceneLoaderPYSON::load(const char *filename)
{
    std::stringstream s;
    s << "from pysonloader import load as pysonload" ;

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

    PyObject* result = PyRun_String(s.str().c_str(), Py_file_input, pDict, pDict);
    if (result==nullptr){
         PyErr_Print();
         return nullptr;
    }

    msg_info("SceneLoaderPYSON") << "Loading file: " << filename ;
    PyRun_String("print(dir())", Py_file_input, pDict, pDict);

    PyObject *pFunc = PyDict_GetItemString(pDict, "pysonload");
    if (PyCallable_Check(pFunc))
    {
        Node::SPtr rootNode = Node::create("root");
        SP_CALL_MODULEFUNC(pFunc, "(Os)", sofa::PythonFactory::toPython(rootNode.get()), filename)
        return rootNode;
    }

    assert(PyCallable_Check(pFunc));
    return nullptr ;
}

void SceneLoaderPYSON::write(Node* node, const char *filename)
{
    msg_error("SceneLoaderPYSON") << "Nothing will work" ;
}


#if 0
sofa::simulation::Node::SPtr SceneLoaderPYSON::loadSceneWithArguments(const char *filename, const std::vector<std::string>& arguments)
{
    if(!OurHeader.empty() && 0 != PyRun_SimpleString(OurHeader.c_str()))
    {
        SP_MESSAGE_ERROR( "header script run error." )
        return NULL;
    }

    PythonEnvironment::runString("createScene=None");
    PythonEnvironment::runString("createSceneAndController=None");

    PythonEnvironment::runString(std::string("__file__=\"") + filename + "\"");

    // We go the the current file's directory so that all relative path are correct
    helper::system::SetDirectory chdir ( filename );

    notifyLoadingScene();
    if(!PythonEnvironment::runFile(helper::system::SetDirectory::GetFileName(filename).c_str(), arguments))
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( "scene script load error." )
        return NULL;
    }

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

    // pFunc is also a borrowed reference
    PyObject *pFunc = PyDict_GetItemString(pDict, "createScene");
    if (PyCallable_Check(pFunc))
    {
        Node::SPtr rootNode = Node::create("root");
        SP_CALL_MODULEFUNC(pFunc, "(O)", sofa::PythonFactory::toPython(rootNode.get()))

        return rootNode;
    }
    else
    {
        PyObject *pFunc = PyDict_GetItemString(pDict, "createSceneAndController");
        if (PyCallable_Check(pFunc))
        {
            Node::SPtr rootNode = Node::create("root");
            SP_CALL_MODULEFUNC(pFunc, "(O)", sofa::PythonFactory::toPython(rootNode.get()))

            rootNode->addObject( core::objectmodel::New<component::controller::PythonMainScriptController>( filename ) );

            return rootNode;
        }
    }

    SP_MESSAGE_ERROR( "cannot create Scene, no \"createScene(rootNode)\" nor \"createSceneAndController(rootNode)\" module method found." )
    return NULL;
}


bool SceneLoaderPYSON::loadTestWithArguments(const char *filename, const std::vector<std::string>& arguments)
{
    if(!OurHeader.empty() && 0 != PyRun_SimpleString(OurHeader.c_str()))
    {
        SP_MESSAGE_ERROR( "header script run error." )
        return false;
    }

    PythonEnvironment::runString("createScene=None");
    PythonEnvironment::runString("createSceneAndController=None");

    PythonEnvironment::runString(std::string("__file__=\"") + filename + "\"");

    // it runs the unecessary SofaPython script but it is not a big deal
    if(!PythonEnvironment::runFile(filename,arguments))
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( "script load error." )
        return false;
    }

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

    // pFunc is also a borrowed reference
    PyObject *pFunc = PyDict_GetItemString(pDict, "run");
    if (PyCallable_Check(pFunc))
    {
        PyObject *res = PyObject_CallObject(pFunc,0);
        printPythonExceptions();

        if( !res )
        {
            SP_MESSAGE_ERROR( "Python test 'run' function does not return any value" )
            return false;
        }
        else if( !PyBool_Check(res) )
        {
            SP_MESSAGE_ERROR( "Python test 'run' function does not return a boolean" )
            Py_DECREF(res);
            return false;
        }

        bool result = ( res == Py_True );
        Py_DECREF(res);

        return result;
    }
    else
    {
        SP_MESSAGE_ERROR( "Python test has no 'run'' function" )
        return false;
    }
}
#endif //




} // namespace _sceneloaderpyson_

} // namespace simulation

} // namespace sofa

