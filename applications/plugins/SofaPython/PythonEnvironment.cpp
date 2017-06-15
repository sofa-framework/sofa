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
#include "PythonEnvironment.h"
#include "PythonScriptController.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/Node.h>

#include <sofa/helper/Utils.h>

#if defined(__linux__)
#  include <dlfcn.h>            // for dlopen(), see workaround in Init()
#endif


using namespace sofa::component::controller;

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

namespace sofa
{

namespace simulation
{

PyMODINIT_FUNC initModulesHelper(const std::string& name, PyMethodDef* methodDef)
{
    Py_InitModule(name.c_str(), methodDef);
}

void PythonEnvironment::addModule(const std::string& name, PyMethodDef* methodDef)
{
    initModulesHelper(name, methodDef);
}

void PythonEnvironment::Init()
{
    std::string pythonVersion = Py_GetVersion();

#ifndef NDEBUG
    SP_MESSAGE_INFO("Python version: " + pythonVersion)
#endif

#if defined(__linux__)
    // WARNING: workaround to be able to import python libraries on linux (like
    // numpy), at least on Ubuntu (see http://bugs.python.org/issue4434). It is
    // not fixing the real problem, but at least it is working for now.
    std::string pythonLibraryName = "libpython" + std::string(pythonVersion,0,3) + ".so";
    dlopen( pythonLibraryName.c_str(), RTLD_LAZY|RTLD_GLOBAL );
#endif

    // Prevent the python terminal from being buffered, not to miss or mix up traces.
    if( putenv( (char*)"PYTHONUNBUFFERED=1" ) )
        SP_MESSAGE_WARNING("failed to set environment variable PYTHONUNBUFFERED")

    if ( !Py_IsInitialized() )
    {
        Py_Initialize();
    }

    // Append sofa modules to the embedded python environment.
    bindSofaPythonModule();

    // Required for sys.path, used in addPythonModulePath().
    PyRun_SimpleString("import sys");

    // Force C locale.
    PyRun_SimpleString("import locale");
    PyRun_SimpleString("locale.setlocale(locale.LC_ALL, 'C')");

    // Workaround: try to import numpy and to launch numpy.finfo to cache data;
    // this prevents a deadlock when calling numpy.finfo from a worker thread.
    // ocarre: may crash on some configurations, we have to find a fix
    PyRun_SimpleString("\
try:\n\
    import numpy\n\
    numpy.finfo(float)\n\
except:\n\
    pass");


    // If the script directory is not available (e.g. if the interpreter is invoked interactively
    // or if the script is read from standard input), path[0] is the empty string,
    // which directs Python to search modules in the current directory first.
    PyRun_SimpleString(std::string("sys.path.insert(0,\"\")").c_str());


    // Add the paths to the plugins' python modules to sys.path.  Those paths
    // are read from all the files in 'etc/sofa/python.d'
    std::string confDir = Utils::getSofaPathPrefix() + "/etc/sofa/python.d";
    if (FileSystem::exists(confDir))
    {
        std::vector<std::string> files;
        FileSystem::listDirectory(confDir, files);
        for (size_t i=0; i<files.size(); i++)
        {
            addPythonModulePathsFromConfigFile(confDir + "/" + files[i]);
        }
    }

    // Add the directories listed in the SOFAPYTHON_PLUGINS_PATH environnement
    // variable (colon-separated) to sys.path
    char * pathVar = getenv("SOFAPYTHON_PLUGINS_PATH");
    if (pathVar != NULL)
    {
        std::istringstream ss(pathVar);
        std::string path;
        while(std::getline(ss, path, ':'))
        {
            if (FileSystem::exists(path))
                addPythonModulePathsForPlugins(path);
            else
                SP_MESSAGE_WARNING("no such directory: '" + path + "'");
        }
    }

    // python livecoding related
    PyRun_SimpleString("from SofaPython.livecoding import onReimpAFile");

    // general sofa-python stuff
    PyRun_SimpleString("import SofaPython");

    // python modules are automatically reloaded at each scene loading
    setAutomaticModuleReload( true );
}

void PythonEnvironment::Release()
{
    // Finish the Python Interpreter
    Py_Finalize();
}

void PythonEnvironment::addPythonModulePath(const std::string& path)
{
    static std::set<std::string> addedPath;
    if (addedPath.find(path)==addedPath.end()) {
        // note not to insert at first 0 place
        // an empty string must be at first so modules can be found in the current directory first.
        PyRun_SimpleString(std::string("sys.path.insert(1,\""+path+"\")").c_str());
        SP_MESSAGE_INFO("Added '" + path + "' to sys.path");
        addedPath.insert(path);
    }
}

void PythonEnvironment::addPythonModulePathsFromConfigFile(const std::string& path)
{
    std::ifstream configFile(path.c_str());
    std::string line;
    while(std::getline(configFile, line))
    {
        if (!FileSystem::isAbsolute(line))
        {
            line = Utils::getSofaPathPrefix() + "/" + line;
        }
        addPythonModulePath(line);
    }
}

void PythonEnvironment::addPythonModulePathsForPlugins(const std::string& pluginsDirectory)
{
    std::vector<std::string> files;
    FileSystem::listDirectory(pluginsDirectory, files);

    for (std::vector<std::string>::iterator i = files.begin(); i != files.end(); ++i)
    {
        const std::string pluginPath = pluginsDirectory + "/" + *i;
        if (FileSystem::exists(pluginPath) && FileSystem::isDirectory(pluginPath))
        {
            const std::string pythonDir = pluginPath + "/python";
            if (FileSystem::exists(pythonDir) && FileSystem::isDirectory(pythonDir))
            {
                addPythonModulePath(pythonDir);
            }
        }
    }
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

// some basic RAII stuff to handle init/termination cleanly
  namespace {

    struct raii {
      raii() {
          // initialization is done when loading the plugin
          // otherwise it can be executed too soon
          // when an application is directly linking with the SofaPython library
//        PythonEnvironment::Init();
      }

      ~raii() {
        PythonEnvironment::Release();
      }

    };

    static raii singleton;
  }

// basic script functions
std::string PythonEnvironment::getError()
{
    std::string error;

    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if(pvalue)
        error = PyString_AsString(pvalue);

    return error;
}

bool PythonEnvironment::runString(const std::string& script)
{
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject* result = PyRun_String(script.data(), Py_file_input, pDict, pDict);

    if(0 == result)
    {
        SP_MESSAGE_ERROR("Script (string) import error")
        PyErr_Print();

        return false;
    }

    Py_DECREF(result);

    return true;
}

bool PythonEnvironment::runFile( const char *filename, const std::vector<std::string>& arguments)
{
//    SP_MESSAGE_INFO( "Loading python script \""<<filename<<"\"" )
    std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename);
    std::string bareFilename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename);
//    SP_MESSAGE_INFO( "script directory \""<<dir<<"\"" )

    //    SP_MESSAGE_INFO( commandString.c_str() )

    if(!arguments.empty())
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

    //  Py_BEGIN_ALLOW_THREADS

    // Load the scene script
    char* pythonFilename = strdup(filename);
    PyObject* scriptPyFile = PyFile_FromString(pythonFilename, (char*)("r"));
    free(pythonFilename);

    if( !scriptPyFile )
    {
        SP_MESSAGE_ERROR("cannot open file:" << filename)
        PyErr_Print();
        return false;
    }

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

    std::string backupFileName;
    PyObject* backupFileObject = PyDict_GetItemString(pDict, "__file__");
    if(backupFileObject)
        backupFileName = PyString_AsString(backupFileObject);

    PyObject* newFileObject = PyString_FromString(filename);
    PyDict_SetItemString(pDict, "__file__", newFileObject);

    int error = PyRun_SimpleFileEx(PyFile_AsFile(scriptPyFile), filename, 1);

    backupFileObject = PyString_FromString(backupFileName.c_str());
    PyDict_SetItemString(pDict, "__file__", backupFileObject);

    //  Py_END_ALLOW_THREADS

    if(0 != error)
    {
        SP_MESSAGE_ERROR("Script (file:" << bareFilename << ") import error")
        PyErr_Print();
        return false;
    }

    return true;
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

void PythonEnvironment::SceneLoaderListerner::rightBeforeLoadingScene()
{
    // unload python modules to force importing their eventual modifications
    PyRun_SimpleString("SofaPython.unloadModules()");
}


void PythonEnvironment::setAutomaticModuleReload( bool b )
{
    if( b )
        SceneLoader::addListener( SceneLoaderListerner::getInstance() );
    else
        SceneLoader::removeListener( SceneLoaderListerner::getInstance() );
}


void PythonEnvironment::excludeModuleFromReload( const std::string& moduleName )
{
    PyRun_SimpleString( std::string( "try: SofaPython.__SofaPythonEnvironment_modulesExcludedFromReload.append('" + moduleName + "')\nexcept:pass" ).c_str() );
}



} // namespace simulation

} // namespace sofa



