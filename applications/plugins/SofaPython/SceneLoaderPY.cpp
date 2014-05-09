/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "PythonEnvironment.h"
#include "SceneLoaderPY.h"
#include "ScriptEnvironment.h"
#include "PythonMacros.h"

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>

namespace sofa
{

namespace simulation
{



bool SceneLoaderPY::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="py");
}

/// get the file type description
std::string SceneLoaderPY::getFileTypeDesc()
{
    return "Python Scenes";
}

/// get the list of file extensions
void SceneLoaderPY::getExtensionList(ExtensionList* list)
{
    list->clear();
   // list->push_back("pyscn");
    list->push_back("py");
}

sofa::simulation::Node::SPtr SceneLoaderPY::load(const char *filename)
{
    return loadSceneWithArguments( filename );
}



sofa::simulation::Node::SPtr SceneLoaderPY::loadSceneWithArguments(const char *filename, const std::vector<std::string>& arguments)
{
    PyObject *script = PythonEnvironment::importScript(filename,arguments);
    if (!script)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( "scene script load error." )
        return NULL;
    }

    // pDict is a borrowed reference
    PyObject *pDict = PyModule_GetDict(script);
    if (!pDict)
    {
        // DICT ERROR
        SP_MESSAGE_ERROR( "script dictionnary load error." )
        return NULL;
    }

    // pFunc is also a borrowed reference
    PyObject *pFunc = PyDict_GetItemString(pDict, "createScene");
    if (PyCallable_Check(pFunc))
    {
        Node::SPtr rootNode = getSimulation()->createNewGraph("root");
        ScriptEnvironment::enableNodeQueuedInit(false);
        SP_CALL_MODULEFUNC(pFunc, "(O)", SP_BUILD_PYSPTR(rootNode.get()))
        ScriptEnvironment::enableNodeQueuedInit(true);
        return rootNode;
    }
    else
    {
        SP_MESSAGE_ERROR( "cannot create Scene, no \"createScene(rootNode)\" module method found." )
    }

    return NULL;
}


bool SceneLoaderPY::loadTestWithArguments(const char *filename, const std::vector<std::string>& arguments)
{
    // it runs the unecessary SofaPython script but it is not a big deal
    PyObject *script = PythonEnvironment::importScript(filename,arguments);
    if (!script)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( "scene script load error." )
        return false;
    }

    // pDict is a borrowed reference
    PyObject *pDict = PyModule_GetDict(script);
    if (!pDict)
    {
        // DICT ERROR
        SP_MESSAGE_ERROR( "script dictionnary load error." )
        return false;
    }

    // pFunc is also a borrowed reference
    PyObject *pFunc = PyDict_GetItemString(pDict, "run");
    if (PyCallable_Check(pFunc))
    {
        ScriptEnvironment::enableNodeQueuedInit(false);

        PyObject *res = PyObject_CallObject(pFunc,0);
        if( !res )
        {
            SP_MESSAGE_ERROR( "Python test has no 'run'' function" )
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


        ScriptEnvironment::enableNodeQueuedInit(true);
        return result;
    }
    else
    {
        SP_MESSAGE_ERROR( "cannot create Scene, no \"createScene(rootNode)\" module method found." )
        return false;
    }
}


} // namespace simulation

} // namespace sofa

