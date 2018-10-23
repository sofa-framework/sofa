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

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/SetDirectory.h>

#include <sofa/helper/Utils.h>
#include <sofa/helper/StringUtils.h>
using sofa::helper::getAStringCopy ;

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

#include <SofaPython3/PythonEnvironment.h>

#include <sofa/helper/logging/Messaging.h>
MSG_REGISTER_CLASS(sofapython3::PythonEnvironment, "SofaPython3::PythonEnvironment")

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory;

//#include "Python.h"

#include <pybind11/embed.h>
namespace py = pybind11;

#include "SceneLoaderPY3.h"
using sofapython3::SceneLoaderPY3;

namespace sofapython3
{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The PythonEnvironmentData class which hold "static" data as long as python is running
///
/// The class currently hold the argv that are exposed in the python 'sys.argv'.
/// The elements added are copied and the object hold the pointer to the memory allocated.
/// The memory is release when the object is destructed or the reset function called.
///
/// Other elements than sys.argv may be added depending on future needs.
///
////////////////////////////////////////////////////////////////////////////////////////////////////
class PythonEnvironmentData
{
public:
    ~PythonEnvironmentData() { reset(); }

    int size() { return m_argv.size(); }

    void add(const std::string& data)
    {
        m_argv.push_back( Py_DecodeLocale(data.c_str(), nullptr) );
    }

    void reset()
    {
        for(auto s : m_argv){
            PyMem_Free(s);
        }
        m_argv.clear();
        addedPath.clear();
    }

    wchar_t* getDataAt(unsigned int index)
    {
        return m_argv[index];
    }

    wchar_t** getDataBuffer()
    {
        return &m_argv[0];
    }

    std::set<std::string> addedPath;
private:
    std::vector<wchar_t*> m_argv;
};

PythonEnvironmentData* PythonEnvironment::getStaticData()
{
    static PythonEnvironmentData* m_staticdata { nullptr } ;

    if( !m_staticdata )
        m_staticdata = new PythonEnvironmentData();

    return m_staticdata;
}

py::module PythonEnvironment::importFromFile(const std::string& module, const std::string& path, py::object& globals)
{
    PythonEnvironment::gil lock{__func__};
    py::dict locals;
    locals["module_name"] = py::cast(module); // have to cast the std::string first
    locals["path"]        = py::cast(path);
    msg_info("SofaPython3") << "Importing module: " << path ;
    py::eval<py::eval_statements>(            // tell eval we're passing multiple statements
                                              "import imp\n"
                                              "new_module = imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))\n",
                                              globals,
                                              locals);
    return py::cast<py::module>(locals["new_module"]);
}


void PythonEnvironment::Init()
{
    std::string pythonVersion = Py_GetVersion();
    msg_info("SofaPython3") << " Initializing with python version " << pythonVersion;

    if( !SceneLoaderFactory::getInstance()->getEntryFileExtension("py3") )
    {
        msg_info("SofaPython3") << "Registering a scene loader for [.py, .py3, .pyscn, .py3scn] files." ;
        SceneLoaderFactory::getInstance()->addEntry(new SceneLoaderPY3());
    }


    #if defined(__linux__)
        // WARNING: workaround to be able to import python libraries on linux (like
        // numpy), at least on Ubuntu (see http://bugs.python.org/issue4434). It is
        // not fixing the real problem, but at least it is working for now.
        // dmarchal: The problem still exists python3 10/10/2018.
        std::string pythonLibraryName = "libpython" + std::string(pythonVersion,0,3) + "m.so";
        dlopen( pythonLibraryName.c_str(), RTLD_LAZY|RTLD_GLOBAL );
        msg_info("SofaPython3") << "Shared library name is '" << pythonLibraryName << "'" ;
    #endif

    /// Prevent the python terminal from being buffered, not to miss or mix up traces.
    if( putenv( (char*)"PYTHONUNBUFFERED=1" ) )
        msg_warning("SofaPython3") << "failed to set environment variable PYTHONUNBUFFERED";

    if ( !Py_IsInitialized() )
    {
        msg_info("SofaPython3") << "Intializing python";
        py::initialize_interpreter();
    }

    PyEval_InitThreads();

    // the first gil lock is here
    gil lock(__func__);

    // Required for sys.path, used in addPythonModulePath().
    PyRun_SimpleString("import sys");

    // Force C locale.
    PyRun_SimpleString("import locale");
    PyRun_SimpleString("locale.setlocale(locale.LC_ALL, 'C')");

    // Workaround: try to import numpy and to launch numpy.finfo to cache data;
    // this prevents a deadlock when calling numpy.finfo from a worker thread.
    PyRun_SimpleString("try:\n\timport numpy;numpy.finfo(float)\nexcept:\n\tpass");

    // Workaround: try to import scipy from the main thread this prevents a deadlock when importing
    // scipy from a worker thread when we use the SofaScene asynchronous loading
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

    /// Add the directories listed in the SOFAPYTHON_PLUGINS_PATH environnement
    /// variable (colon-separated) to sys.path
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
                msg_warning("SofaPython3") << "no such directory: '" + path + "'";
        }
    }

    // python livecoding related
    //PyRun_SimpleString("from SofaPython3.livecoding import onReimpAFile");

    // general sofa-python stuff

    // python modules are automatically reloaded at each scene loading
    //setAutomaticModuleReload( true );
}

void PythonEnvironment::Release()
{
    /// Finish the Python Interpreter
    /// obviously can't use raii here
    if(  Py_IsInitialized() ) {
        PyGILState_Ensure();
        py::finalize_interpreter();
        getStaticData()->reset();
    }

}

void PythonEnvironment::addPythonModulePath(const std::string& path)
{
    PythonEnvironmentData* data = getStaticData() ;
    if (  data->addedPath.find(path)==data->addedPath.end()) {
        // note not to insert at first 0 place
        // an empty string must be at first so modules can be found in the current directory first.

        {
            gil lock(__func__);
            PyRun_SimpleString(std::string("sys.path.insert(1,\""+path+"\")").c_str());
        }

        msg_info("SofaPython3") << "Added '" + path + "' to sys.path";
        data->addedPath.insert(path);
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
namespace
{

struct raii {
    raii() {
        /// initialization is done when loading the plugin
        /// otherwise it can be executed too soon
        /// when an application is directly linking with the SofaPython library
    }

    ~raii() {
        //PythonEnvironment::Release();
    }
};

static raii singleton;

} // namespace


/// basic script functions
std::string PythonEnvironment::getError()
{
    gil lock(__func__);
    std::string error;

    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if(pvalue)
        error = PyBytes_AsString(pvalue);

    return error;
}

bool PythonEnvironment::runString(const std::string& script)
{
    gil lock(__func__);
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject* result = PyRun_String(script.data(), Py_file_input, pDict, pDict);

    if(0 == result)
    {
        msg_error("SofaPython3") << "Script (string) import error";
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
        std::string tmp=PyBytes_AsString(PyObject_Str(res));
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
        std::string tmp=PyBytes_AsString(PyObject_Str(res));
        Py_DECREF(res) ;
        return tmp;
    }
    return "Python Stack is empty.";
}

sofa::helper::logging::FileInfo::SPtr PythonEnvironment::getPythonCallingPointAsFileInfo()
{
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("SofaPython"));
    PyObject* pFunc = PyDict_GetItemString(pDict, "getPythonCallingPoint");
    if (pFunc && PyCallable_Check(pFunc))
    {
        PyObject* res = PyObject_CallFunction(pFunc, nullptr);
        if(res && PySequence_Check(res) ){
            PyObject* filename = PySequence_GetItem(res, 0) ;
            PyObject* number = PySequence_GetItem(res, 1) ;
            std::string tmp=PyBytes_AsString(filename);
            auto lineno = PyLong_AsLong(number);
            Py_DECREF(res) ;
            return SOFA_FILE_INFO_COPIED_FROM(tmp, lineno);
        }
    }
    return SOFA_FILE_INFO_COPIED_FROM("undefined", -1);
}

void PythonEnvironment::setArguments(const std::string& filename, const std::vector<std::string>& arguments)
{
    gil lock(__func__);
    const std::string basename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename.c_str());

    PythonEnvironmentData* data = getStaticData() ;
    data->reset() ;
    data->add( basename );

    if(!arguments.empty()) {
        for(const std::string& arg : arguments) {
            data->add(arg);
        }
    }

    PySys_SetArgvEx( data->size(), data->getDataBuffer(), 0);
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

} // namespace sofapython



