/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/helper/StringUtils.h>
using sofa::helper::getAStringCopy ;
#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;
using sofa::helper::system::Plugin;

#if defined(__linux__)
#  include <dlfcn.h>            // for dlopen(), see workaround in Init()
#endif

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

namespace sofa
{

namespace simulation
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
        m_argv.push_back( getAStringCopy(data.c_str()) );
    }

    void reset()
    {
        for(auto s : m_argv)
            delete[] s;
        m_argv.clear();
    }

    char* getDataAt(unsigned int index)
    {
        return m_argv[index];
    }

    char** getDataBuffer()
    {
        return &m_argv[0];
    }

private:
    std::vector<char*> m_argv;
};

PythonEnvironmentData* PythonEnvironment::getStaticData()
{
    static PythonEnvironmentData* m_staticdata { nullptr } ;

    if( !m_staticdata )
        m_staticdata = new PythonEnvironmentData();

    return m_staticdata;
}

std::string PythonEnvironment::pluginLibraryPath = "";

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
    std::vector< std::string > paths = sofa::helper::system::DataRepository.getPaths();
    paths.push_back(Utils::getSofaPathPrefix());
    for (auto path : paths)
    {
        std::string confDir = path + "/etc/sofa/python.d";
        if (FileSystem::exists(confDir))
        {
            std::vector<std::string> files;
            FileSystem::listDirectory(confDir, files);

            for (size_t i = 0; i < files.size(); i++)
            {
                addPythonModulePathsFromConfigFile(confDir + "/" + files[i]);
            }
        }
    }

    // Add the directories listed in the SOFAPYTHON_PLUGINS_PATH environnement
    // variable (colon-separated) to sys.path
    char * pathVar = getenv("SOFAPYTHON_PLUGINS_PATH");
    if (pathVar != nullptr)
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

    // Add the directories listed in SofaPython.so/../../python
    addPythonModulePathsForPluginsByName("SofaPython");

    // python livecoding related
    PyRun_SimpleString("from SofaPython.livecoding import onReimpAFile");

    // general sofa-python stuff
    PyRun_SimpleString("import SofaPython");

    // python modules are automatically reloaded at each scene loading
    setAutomaticModuleReload( true );

    // Initialize pluginLibraryPath by reading PluginManager's map
    std::map<std::string, Plugin>& map = PluginManager::getInstance().getPluginMap();
    for( const auto& elem : map)
    {
        Plugin p = elem.second;
        if ( p.getModuleName() == sofa_tostring(SOFA_TARGET) )
        {
            pluginLibraryPath = elem.first;
        }
    }
}

void PythonEnvironment::Release()
{
    removePluginManagerCallback();

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
    bool added = false;

    std::vector<std::string> pythonDirs = {
        pluginsDirectory + "/python",
        pluginsDirectory + "/python2.7",
    };

    std::vector<std::string> files;
    FileSystem::listDirectory(pluginsDirectory, files);
    for (std::vector<std::string>::iterator i = files.begin(); i != files.end(); ++i)
    {
        const std::string pluginSubdir = pluginsDirectory + "/" + *i;
        if (FileSystem::exists(pluginSubdir) && FileSystem::isDirectory(pluginSubdir))
        {
            pythonDirs.push_back(pluginSubdir + "/python");
            pythonDirs.push_back(pluginSubdir + "/python2.7");
        }
    }

    for(std::string pythonDir : pythonDirs)
    {
        // Search for a subdir "site-packages"
        if (FileSystem::exists(pythonDir+"/site-packages") && FileSystem::isDirectory(pythonDir+"/site-packages"))
        {
            addPythonModulePath(pythonDir+"/site-packages");
            added = true;
        }
        // Or fallback to "python"
        else if (FileSystem::exists(pythonDir) && FileSystem::isDirectory(pythonDir))
        {
            addPythonModulePath(pythonDir);
            added = true;
        }
    }

    if(!added)
    {
        msg_info("PythonEnvironment") << "No python dir found in " << pluginsDirectory;
    }
}

void PythonEnvironment::addPythonModulePathsForPluginsByName(const std::string& pluginName)
{
    std::map<std::string, Plugin>& map = PluginManager::getInstance().getPluginMap();
    for( const auto& elem : map)
    {
        Plugin p = elem.second;
        if ( p.getModuleName() == pluginName )
        {
            std::string pluginLibraryPath = elem.first;
            // moduleRoot should be 2 levels above the library (plugin_name/lib/plugin_name.so)
            std::string moduleRoot = FileSystem::getParentDirectory(FileSystem::getParentDirectory(pluginLibraryPath));

            addPythonModulePathsForPlugins(moduleRoot);
            return;
        }
    }
    msg_info("PythonEnvironment") << pluginName << " not found in PluginManager's map.";
}

void PythonEnvironment::addPluginManagerCallback()
{
    PluginManager::getInstance().addOnPluginLoadedCallback(pluginLibraryPath,
        [](const std::string& pluginLibraryPath, const Plugin& plugin) {
            // WARNING: loaded plugin must be organized like plugin_name/lib/plugin_name.so
            for ( auto path : sofa::helper::system::PluginRepository.getPaths() )
            {
                std::string pluginRoot = FileSystem::cleanPath( path + "/" + plugin.getModuleName() );
                if ( FileSystem::exists(pluginRoot) && FileSystem::isDirectory(pluginRoot) )
                {
                    addPythonModulePathsForPlugins(pluginRoot);
                    return;
                }
            }
        }
    );
}

void PythonEnvironment::removePluginManagerCallback()
{
    PluginManager::getInstance().removeOnPluginLoadedCallback(pluginLibraryPath);
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

    if(nullptr == result)
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

void PythonEnvironment::setArguments(const std::string& filename, const std::vector<std::string>& arguments)
{
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

bool PythonEnvironment::runFile(const std::string& filename, const std::vector<std::string>& arguments) {
    const gil lock(__func__);
    const std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());

    const std::string basename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename.c_str());

    // Load the scene script
    PyObject* script = PyFile_FromString((char*)filename.c_str(), (char*)("r"));

    if(!arguments.empty())
        setArguments(filename, arguments);

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
        PyObject* __tmpfile__ = PyString_FromString(filename.c_str());
        PyDict_SetItemString(__main__, "__file__", __tmpfile__);
        Py_XDECREF(__tmpfile__);
    }

    const int error = PyRun_SimpleFileEx(PyFile_AsFile(script), filename.c_str(), 0);

    // don't wait for gc to close the file
    PyObject_CallMethod(script, (char*) "close", nullptr);
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



