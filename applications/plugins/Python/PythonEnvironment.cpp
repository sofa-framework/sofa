#include "PythonEnvironment.h"
#include "PythonBindings.h"

#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/component/contextobject/CoordinateSystem.h>

#include <boost/python.hpp>
using namespace boost::python;

#include "PythonScriptController.h"
using namespace sofa::component::controller;

using namespace sofa::simulation::tree;

namespace sofa
{

namespace simulation
{


static bool     m_Initialized = false;



void PythonEnvironment::Init()
{
    if (m_Initialized) return;
    // Initialize the Python Interpreter
    std::cout<<"<PYTHON> Initializing python framework..."<<std::endl;
    std::cout<<"<PYTHON> "<<Py_GetVersion()<<std::endl;
//    PyEval_InitThreads();
    Py_Initialize();
    std::cout<<"<PYTHON> Registering Sofa bindings..."<<std::endl;


    // append sofa modules to the embedded python environment
    registerSofaPythonModule();

    std::cout<<"<PYTHON> Initialization done."<<std::endl;



    m_Initialized = true;

}

void PythonEnvironment::Release()
{
    if (!m_Initialized) return;
    // Finish the Python Interpreter
    Py_Finalize();

    m_Initialized = false;
}

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




// basic script functions
PyObject* PythonEnvironment::importScript( const char *filename )
{
    Init(); // MUST be called at least once; so let's call it each time we load a python script

    std::cout << "<PYTHON> Loading python script \""<<filename<<"\""<<std::endl;
    std::string dir = sofa::helper::system::SetDirectory::GetParentDir(filename);
    std::string bareFilename = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename);
    //  std::cout << "<PYTHON> script directory \""<<dir<<"\""<<std::endl;

    // temp: directory always added to environment;
    // TODO: check if the path is already set to this directory...

    // append current path to Python module search path...
    std::string commandString = "sys.path.append(\""+dir+"\")";

    //  printf("<PYTHON> %s\n",commandString.c_str());

    PyObject *pModule = 0;

    //  Py_BEGIN_ALLOW_THREADS

    PyRun_SimpleString("import sys");
    //  printf("<PYTHON> 1\n");
    PyRun_SimpleString(commandString.c_str());
    //  printf("<PYTHON> 2\n");

    // Load the module object
    pModule = PyImport_Import(PyString_FromString(bareFilename.c_str()));
    //  printf("<PYTHON> 3\n");

    //  Py_END_ALLOW_THREADS

    if (!pModule)
    {
        printf("<PYTHON> Script \"%s\" import error\n",bareFilename.c_str());
        PyErr_Print();
        return 0;
    }
    //  printf("<PYTHON> 5\n");

    return pModule;
}

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
            printf("<PYTHON> exception\n");
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

} // namespace core

} // namespace sofa



