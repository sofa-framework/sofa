/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
* Contributors:                                                               *
*    - damien.marchal@univ-lille1.fr Copyright (C) CNRS                       *
*******************************************************************************/
#include <sstream>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationCommon/xml/NodeElement.h>
#include <SofaSimulationCommon/FindByTypeVisitor.h>

#include <SofaPython/Binding_Node.h>
#include <SofaPython/PythonFactory.h>
#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonEnvironment.h>
#include <SofaPython/PythonToSofa.inl>
#include "SceneLoaderPSL.h"

using namespace sofa::core::objectmodel;

namespace sofa
{

namespace simulation
{

namespace _sceneloaderpsl_
{

bool SceneLoaderPSL::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="psl" || ext=="pslx" || ext == "pslp");
}

bool SceneLoaderPSL::canWriteFileExtension(const char *extension)
{
    return canLoadFileExtension(extension);
}

/// get the file type description
std::string SceneLoaderPSL::getFileTypeDesc()
{
    return "Pythonized Scene Language";
}

/// get the list of file extensions
void SceneLoaderPSL::getExtensionList(ExtensionList* list)
{
    list->clear();
    list->push_back("psl");  /// Human-JSON version
    list->push_back("pslx"); /// XML version
    list->push_back("pslp"); /// Python Pickled verson
}

void SceneLoaderPSL::write(sofa::simulation::Node* n, const char *filename)
{
    PythonEnvironment::gil lock(__func__);

    std::stringstream s;
    s << "from pslloader import save as pslsave" ;

    msg_info("SceneLoaderPSL") << "Saving file: " << filename ;

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

    PyObject* result = PyRun_String(s.str().c_str(), Py_file_input, pDict, pDict);
    if (result==nullptr){
         PyErr_Print();
         return;
    }

    PyObject *pFunc = PyDict_GetItemString(pDict, "pslsave");
    if (PyCallable_Check(pFunc))
    {
        Node::SPtr rootNode = Node::create("root");
        SP_CALL_MODULEFUNC(pFunc, "(Os)", sofa::PythonFactory::toPython(n), filename)
        return;
    }

    assert(PyCallable_Check(pFunc));
    return;
}


sofa::simulation::Node::SPtr SceneLoaderPSL::load(const char *filename)
{
    PythonEnvironment::gil lock(__func__);

    std::stringstream s ;
    s << "from pslloader import load as pslload" ;

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__")) ;

    PyObject* result = PyRun_String(s.str().c_str(), Py_file_input, pDict, pDict) ;
    if (result==nullptr){
         PyErr_Print() ;
         return nullptr ;
    }

    msg_info("SceneLoaderPSL") << "Loading file: " << filename ;

    PyObject *pFunc = PyDict_GetItemString(pDict, "pslload") ;
    if (PyCallable_Check(pFunc))
    {
        PyObject *res = PyObject_CallObject(pFunc,Py_BuildValue("(s)", filename));

        /// Check if an exception happens. This is indicated by the return pointer to be equal
        /// to nullptr. If this happens, we check if this is an exit signal or a normal exception.
        /// If it is a "normal" exception we print the stack using the currently installed handler.
        /// In Sofa this handler is sending an error message with the python stack.
        if (!res) {
            if(PyErr_ExceptionMatches(PyExc_SystemExit))  {
                PyErr_Clear();
                throw sofa::simulation::PythonEnvironment::system_exit();
            }
            PyErr_Print();
            return nullptr ;
        }

        /// We check if the returned object is of type Node (which is the only valid case. If this
        /// happens then we get the Sofa pointer to this Node and return it.
        if(PyObject_IsInstance(res, reinterpret_cast<PyObject*>(&SP_SOFAPYTYPEOBJECT(Node)))){
            Node::SPtr rootNode = sofa::py::unwrap<Node>(res) ;
            Py_DECREF(res);
            return rootNode ;
        }

        msg_error("SceneLoaderPSL") << "The loading does not returns a Node object while it should." ;
        Py_DECREF(res);
        return nullptr ;
    }

    return nullptr ;
}


} // namespace _sceneloaderpyson_

} // namespace simulation

} // namespace sofa

