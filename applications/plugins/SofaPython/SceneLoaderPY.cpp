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
#include "SceneLoaderPY.h"
#include "PythonEnvironment.h"
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
    PyObject *script = PythonEnvironment::importScript(filename);
    if (!script)
    {
        // LOAD ERROR
        std::cerr << "<SofaPython> ERROR : scene script load error." << std::endl;
        return NULL;
    }

    // pDict is a borrowed reference
    PyObject *pDict = PyModule_GetDict(script);
    if (!pDict)
    {
        // DICT ERROR
        std::cerr << "<SofaPython> script dictionnary load error." << std::endl;
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
        std::cerr << "<SofaPython> cannot create Scene, no \"createScene(rootNode)\" module method found." << std::endl;
    }

    return NULL;
}



} // namespace simulation

} // namespace sofa

