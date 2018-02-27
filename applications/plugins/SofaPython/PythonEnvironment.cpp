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
#include <fstream>
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

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

namespace sofa
{

namespace simulation
{

PyMODINIT_FUNC initModulesHelper(const std::string& name, PyMethodDef* methodDef)
{
    PythonEnvironment::gil lock(__func__);
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
        SP_MESSAGE_WARNING("failed to set environment variable PYTHONUNBUFFERED");

    if ( !Py_IsInitialized() )
    {
        Py_Initialize();
    }
    
    PyEval_InitThreads();
    
    // the first gil lock is here
    gil lock(__func__);

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
    PyRun_SimpleString("try:\n\timport numpy;numpy.finfo(float)\nexcept:\n\tpass");
    // Workaround: try to import scipy from the main thread this prevents a deadlock when importing scipy from a worker thread when we use the SofaScene asynchronous loading
    PyRun_SimpleString("try:\n\tfrom scipy import misc, optimize\nexcept:\n\tpass\n");


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

    // obviously can't use raii here
    if( Py_IsInitialized() ) {
        PyGILState_Ensure();    
        Py_Finalize();
    }
}

void PythonEnvironment::addPythonModulePath(const std::string& path)
{
    static std::set<std::string> addedPath;
    if (addedPath.find(path)==addedPath.end()) {
        // note not to insert at first 0 place
        // an empty string must be at first so modules can be found in the current directory first.

        {
            gil lock(__func__);
            PyRun_SimpleString(std::string("sys.path.insert(1,\""+path+"\")").c_str());
        }
        
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

// some basic RAII stuff to handle init/termination cleanly
  namespace {

    struct raii {
      raii() {
          // initialization is done when loading the plugin
          // otherwise it can be executed too soon
          // when an application is directly linking with the SofaPython library
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
    gil lock(__func__);
    std::string error;

    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if(pvalue)
        error = PyString_AsString(pvalue);

    return error;
}

bool PythonEnvironment::runString(const std::string& script)
{
    gil lock(__func__);
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

std::string PythonEnvironment::getStackAsString()
{
    gil lock(__func__);
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("SofaPython"));
    PyObject* pFunc = PyDict_GetItemString(pDict, "getStackForSofa");
    if (PyCallable_Check(pFunc))
    {
        PyObject* res = PyObject_CallFunction(pFunc, nullptr);
        std::string tmp=PyString_AsString(PyObject_Str(res));
        Py_DECREF(res) ;
        return tmp;
    }
    return "Python Stack is empty.";
}

std::string PythonEnvironment::getPythonCallingPointString()
{
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("SofaPython"));
    PyObject* pFunc = PyDict_GetItemString(pDict, "getPythonCallingPointAsString");
    if (PyCallable_Check(pFunc))
    {
        PyObject* res = PyObject_CallFunction(pFunc, nullptr);
        std::string tmp=PyString_AsString(PyObject_Str(res));
        Py_DECREF(res) ;
        return tmp;
    }
    return "Python Stack is empty.";
}

helper::logging::FileInfo::SPtr PythonEnvironment::getPythonCallingPointAsFileInfo()
{
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("SofaPython"));
    PyObject* pFunc = PyDict_GetItemString(pDict, "getPythonCallingPoint");
    if (pFunc && PyCallable_Check(pFunc))
    {
        PyObject* res = PyObject_CallFunction(pFunc, nullptr);
        if(res && PySequence_Check(res) ){
            PyObject* filename = PySequence_GetItem(res, 0) ;
            PyObject* number = PySequence_GetItem(res, 1) ;
            std::string tmp=PyString_AsString(filename);
            auto lineno = PyInt_AsLong(number);
            Py_DECREF(res) ;
            return SOFA_FILE_INFO_COPIED_FROM(tmp, lineno);
        }
    }
    return SOFA_FILE_INFO_COPIED_FROM("undefined", -1);
}

bool PythonEnvironment::runFile( const char *filename, const std::vector<std::string>& arguments) {
    const gil lock(__func__);
    const std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename);

    // pro-tip: FileNameWithoutExtension == basename
    const std::string basename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename);

    // setup sys.argv if needed
    if(!arguments.empty() ) {
        std::vector<const char*> argv;
        argv.push_back(basename.c_str());
        
        for(const std::string& arg : arguments) {
            argv.push_back(arg.c_str());
        }
        
        Py_SetProgramName((char*) argv[0]); // TODO check what it is doing exactly
        PySys_SetArgv(argv.size(), (char**)argv.data());
    }
    
    // Load the scene script
    PyObject* script = PyFile_FromString((char*)filename, (char*)("r"));
    
    if( !script ) {
        SP_MESSAGE_ERROR("cannot open file:" << filename)
        PyErr_Print();
        return false;
    }
    
    PyObject* __main__ = PyModule_GetDict(PyImport_AddModule("__main__"));

    // save/restore __main__.__file__
    PyObject* __file__ = PyDict_GetItemString(__main__, "__file__");
    Py_XINCREF(__file__);
    
    // temporarily set __main__.__file__ = filename during file loading
    {
        PyObject* __tmpfile__ = PyString_FromString(filename);
        PyDict_SetItemString(__main__, "__file__", __tmpfile__);
        Py_XDECREF(__tmpfile__);
    }
    
    const int error = PyRun_SimpleFileEx(PyFile_AsFile(script), filename, 0);
    
    // don't wait for gc to close the file
    PyObject_CallMethod(script, (char*) "close", NULL);
    Py_XDECREF(script);
    
    // restore backup if needed
    if(__file__) {
        PyDict_SetItemString(__main__, "__file__", __file__);
    } else {
        const int err = PyDict_DelItemString(__main__, "__file__");
        assert(!err); (void) err;
    }

    Py_XDECREF(__file__);  
    
    if(error) {
        SP_MESSAGE_ERROR("Script (file:" << basename << ") import error")
        PyErr_Print();
        return false;
    }

    return true;
}

void PythonEnvironment::SceneLoaderListerner::rightBeforeLoadingScene()
{
    gil lock(__func__);
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
    gil lock(__func__);    
    PyRun_SimpleString( std::string( "try: SofaPython.__SofaPythonEnvironment_modulesExcludedFromReload.append('" + moduleName + "')\nexcept:pass" ).c_str() );
}



static const bool debug_gil = false;

static PyGILState_STATE lock(const char* trace) {
    if(debug_gil && trace) {
        std::clog << ">> " << trace << " wants the gil" << std::endl;
    }
    
    // this ensures that we start with no active thread before first locking the
    // gil: this way the last gil unlock lets python threads to run (otherwise
    // the main thread still holds the gil, preventing python threads to run
    // until the main thread exits).

    // the first gil aquisition should happen right after the python interpreter
    // is initialized.
    static const PyThreadState* init = PyEval_SaveThread(); (void) init;

    return PyGILState_Ensure();
}


PythonEnvironment::gil::gil(const char* trace)
    : state(lock(trace)),
      trace(trace) { }


PythonEnvironment::gil::~gil() {

    PyGILState_Release(state);
    
    if(debug_gil && trace) {
        std::clog << "<< " << trace << " released the gil" << std::endl;
    }
    
}



PythonEnvironment::no_gil::no_gil(const char* trace)
    : state(PyEval_SaveThread()),
      trace(trace) {
    if(debug_gil && trace) {
        std::clog << ">> " << trace << " temporarily released the gil" << std::endl;
    }
}

PythonEnvironment::no_gil::~no_gil() {

    if(debug_gil && trace) {
        std::clog << "<< " << trace << " wants to reacquire the gil" << std::endl;
    }
    
    PyEval_RestoreThread(state);
}

} // namespace simulation

} // namespace sofa



