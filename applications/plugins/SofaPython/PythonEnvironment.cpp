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
#include "PythonMacros.h"


#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>

#include "PythonScriptController.h"

using namespace sofa::component::controller;

//using namespace sofa::simulation::tree;


// WARNING workaround to be able to import python libraries on linux (like numpy) at least on ubuntu
// more details on http://bugs.python.org/issue4434
// It is not fixing the real problem, but at least it is working for now
#if __linux__
#include <dlfcn.h>
#endif

namespace sofa
{

namespace simulation
{


static bool m_Initialized = false;



void PythonEnvironment::Init()
{
    if (m_Initialized) return;
    // Initialize the Python Interpreter
    // SP_MESSAGE_INFO( "Initializing python framework..." )

    std::string pythonVersion = Py_GetVersion();
    SP_MESSAGE_INFO( "Python framework version: "<<pythonVersion )
//    PyEval_InitThreads();

#if __linux__
    // fixing the library import on ubuntu
    std::string pythonLibraryName = "libpython" + std::string(pythonVersion,0,3) + ".so";
    dlopen( pythonLibraryName.c_str(), RTLD_LAZY|RTLD_GLOBAL );
#endif

    // force python terminal no to be buffered not to miss or mix-up traces
    if( putenv((char*)"PYTHONUNBUFFERED=1") )
        SP_MESSAGE_WARNING( "failed to define environment variable PYTHONUNBUFFERED" )


    //SP_MESSAGE_INFO( "Py_Initialize()" )
    Py_Initialize();
    //SP_MESSAGE_INFO( "Registering Sofa bindings..." )


    // append sofa modules to the embedded python environment
    bindSofaPythonModule();

    // load a python script which search for python packages defined in the modules
//    std::string scriptPy = std::string(SOFA_SRC_DIR) + "/applications/plugins/SofaPython/SofaPython.py";
//
//
//#ifdef WIN32
//    char* scriptPyChar = (char*) malloc(scriptPy.size()*sizeof(char));
//    strcpy(scriptPyChar,scriptPy.c_str());
//    PyObject* PyFileObject = PyFile_FromString(scriptPyChar, "r");
//    PyRun_SimpleFileEx(PyFile_AsFile(PyFileObject), scriptPyChar, 1);
//    free(scriptPyChar);
//#else
//    FILE* scriptPyFile = fopen(scriptPy.c_str(),"r");
//    PyRun_SimpleFile(scriptPyFile, scriptPy.c_str());
//    fclose(scriptPyFile);
//#endif 

    //SP_MESSAGE_INFO( "Initialization done." )

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
PyObject* PythonEnvironment::importScript( const char *filename, const std::vector<std::string>& arguments )
{
    Init(); // MUST be called at least once; so let's call it each time we load a python script

//    SP_MESSAGE_INFO( "Loading python script \""<<filename<<"\"" )
    std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename);
    std::string bareFilename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename);
//    SP_MESSAGE_INFO( "script directory \""<<dir<<"\"" )

    // temp: directory always added to environment;
    // TODO: check if the path is already set to this directory...

    // append current path to Python module search path...
    std::string commandString = "sys.path.append(\""+dir+"\")";

//    SP_MESSAGE_INFO( commandString.c_str() )

    if( !arguments.empty() )
    {
        char**argv = new char*[arguments.size()+1];
        argv[0] = new char[bareFilename.size()+1];
        strcpy( argv[0], bareFilename.c_str() );
        for( size_t i=0 ; i<arguments.size() ; ++i )
        {
            argv[i+1] = new char[arguments[i].size()+1];
            strcpy( argv[i+1], arguments[i].c_str() );
        }

        Py_SetProgramName(argv[0]); // TODO check what it is doing exactly

        PySys_SetArgv(arguments.size()+1, argv);

        for( size_t i=0 ; i<arguments.size()+1 ; ++i )
        {
            delete [] argv[i];
        }
        delete [] argv;
    }

    PyObject *pModule = 0;

    //  Py_BEGIN_ALLOW_THREADS

    PyObject *pSysModuleDict = PyImport_GetModuleDict();

    assert(pSysModuleDict != 0 && PyMapping_Check(pSysModuleDict));

    bool previously_loaded = (PyMapping_HasKey(pSysModuleDict,PyString_FromString(bareFilename.c_str())) == 1);
    /// if true, a module with similar name has been loaded. We need to reload the module.

    PyRun_SimpleString("import sys");
    PyRun_SimpleString(commandString.c_str());

    // Load the module object
    pModule = PyImport_Import(PyString_FromString(bareFilename.c_str()));

    //  Py_END_ALLOW_THREADS

    if (!pModule)
    {
        SP_MESSAGE_ERROR( "Script \""<<bareFilename<<"\" import error" )
        PyErr_Print();
        return 0;
    }

    if (previously_loaded){
        //SP_MESSAGE_INFO( "Script \""<<bareFilename<<"\" reloaded" )
        pModule = PyImport_ReloadModule(pModule);
    }


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
            SP_MESSAGE_EXCEPTION("")
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



