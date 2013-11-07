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
#include "PythonEnvironment.h"


#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/component/contextobject/CoordinateSystem.h>

#include "PythonScriptController.h"
using namespace sofa::component::controller;

//using namespace sofa::simulation::tree;

namespace sofa
{

namespace simulation
{


static bool     m_Initialized = false;



void PythonEnvironment::Init()
{
    if (m_Initialized) return;
    // Initialize the Python Interpreter
    //std::cout<<"<SofaPython> Initializing python framework..."<<std::endl;
    std::cout<<"<SofaPython> Python framework version: "<<Py_GetVersion()<<std::endl;
//    PyEval_InitThreads();
    //std::cout<<"<SofaPython> Py_Initialize();"<<std::endl;
    Py_Initialize();
    //std::cout<<"<SofaPython> Registering Sofa bindings..."<<std::endl;

    // append sofa modules to the embedded python environment
    bindSofaPythonModule();

    //std::cout<<"<SofaPython> Initialization done."<<std::endl;

    m_Initialized = true;

}

void PythonEnvironment::Release()
{
    if (!m_Initialized) return;
    // Finish the Python Interpreter
    Py_Finalize();

    m_Initialized = false;
}
/*
// helper functions
sofa::simulation::tree::GNode::SPtr PythonEnvironment::initGraphFromScript( const char *filename )
{
    PyObject *script = importScript(filename);
    if (!script)
        return 0;

    // the root node
    GNode::SPtr groot = sofa::core::objectmodel::New<GNode>(); // TODO: passer par une factory
    groot->setName( "root" );
   // groot->setGravity( Coord3(0,-10,0) );

    if (!initGraph(script,groot))
        groot = 0;

   else
        printf("Root node name after pyhton: %s\n",groot->getName().c_str());

    Py_DECREF(script);

    return groot;
}
*/



// basic script functions
PyObject* PythonEnvironment::importScript( const char *filename )
{
    Init(); // MUST be called at least once; so let's call it each time we load a python script

//    std::cout << "<SofaPython> Loading python script \""<<filename<<"\""<<std::endl;
    std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename);
    std::string bareFilename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename);
//    std::cout << "<SofaPython> script directory \""<<dir<<"\""<<std::endl;

    // temp: directory always added to environment;
    // TODO: check if the path is already set to this directory...

    // append current path to Python module search path...
    std::string commandString = "sys.path.append(\""+dir+"\")";

//    printf("<SofaPython> %s\n",commandString.c_str());

    PyObject *pModule = 0;

    //  Py_BEGIN_ALLOW_THREADS

    PyObject *pSysModuleDict = PyImport_GetModuleDict();

    assert(pSysModuleDict != 0 && PyMapping_Check(pSysModuleDict));

    bool previously_loaded = (PyMapping_HasKey(pSysModuleDict,PyString_FromString(bareFilename.c_str())) == 1);
    /// if true, a module with similar name has been loaded. We need to do a reload the module.

    PyRun_SimpleString("import sys");
//    printf("<SofaPython> 1\n");
    PyRun_SimpleString(commandString.c_str());
//    printf("<SofaPython> 2\n");

    // Load the module object
    pModule = PyImport_Import(PyString_FromString(bareFilename.c_str()));
    //  printf("<SofaPython> 3\n");

    //  Py_END_ALLOW_THREADS

    if (!pModule)
    {
        printf("<SofaPython> Script \"%s\" import error\n",bareFilename.c_str());
        PyErr_Print();
        return 0;
    }

    if (previously_loaded){
        //printf("<SofaPython> Script \"%s\" reloaded\n",bareFilename.c_str());
        pModule = PyImport_ReloadModule(pModule);
    }

    //    printf("<SofaPython> 5\n");

    return pModule;
}

/*
bool PythonEnvironment::initGraph(PyObject *script, sofa::simulation::tree::GNode::SPtr graphRoot)  // calls the method "initGraph(root)" of the script
{
    // pDict is a borrowed reference
    PyObject *pDict = PyModule_GetDict(script);

    // pFunc is also a borrowed reference
    PyObject *pFunc = PyDict_GetItemString(pDict, "initGraph");

    if (PyCallable_Check(pFunc))
    {
      //  PyObject *args = PyTuple_New(1);
      //  PyTuple_SetItem(args,0,object(graphRoot.get()).ptr());

        try
        {
            //PyObject_CallObject(pFunc, NULL);//args);
            boost::python::call<int>(pFunc,boost::ref(*graphRoot.get()));
        }
        catch (const error_already_set e)
        {
            printf("<SofaPython> exception\n");
            PyErr_Print();

        }

      //  Py_DECREF(args);

        return true;
    }
    else
    {
        PyErr_Print();
        return false;
    }
}
*/

} // namespace simulation

} // namespace sofa



