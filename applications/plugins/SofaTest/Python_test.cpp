#include <SofaPython/PythonScriptEvent.h>
#include <fstream>

#include "Python_test.h"

#include <sofa/helper/system/PluginManager.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>

#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonFactory.h>

namespace sofa {



struct fail {
    const char* message;
    fail(const char* message)
        : message(message) { }
};

static PyObject* operator||(PyObject* obj, const fail& error) {
    if(obj) return obj;
    throw std::runtime_error(error.message);
    return nullptr;
}

static void operator||(int code, const fail& error) {
    if(code >= 0) return;
    throw std::runtime_error(error.message);
}


enum flag : char {
    stop = 1 << 1,
    test_failure = 1 << 2,
    python_error = 1 << 3
};


PyObject* default_excepthook {nullptr} ;

// TODO FIXME: there's probably a MEMLEAK hiding in there, figure it out
static PyObject* except_hook(PyObject* self, PyObject* args) {
    // raise the stop flag
    char* flags = reinterpret_cast<char*>(PyCapsule_GetPointer(self, NULL));
    assert(flags && "cannot get flags pointer (wtf?)");
    if(!flags) std::exit(1);

    *flags |= flag::stop;

    // switch on exception type
    PyObject* type;
    PyObject* value;
    PyObject* traceback;

    if( !PyArg_ParseTuple(args, "OOO", &type, &value, &traceback) ) {
        assert(false && "cannot parse excepthook args (wtf?)");
        std::exit(1);
    }

    if( type == PyExc_AssertionError ) {
        // test failure
        *flags |= flag::test_failure;
    } else {
        // other python error
        *flags |= flag::python_error;
    }

    // call default excepthook to get traceback etc
    return PyObject_CallObject(default_excepthook, args);
}

static PyMethodDef except_hook_def = {
    "sofa_excepthook",
    except_hook,
    METH_VARARGS,
    NULL
};


static void install_sys_excepthook(char* flags) {
    if(!default_excepthook){
        sofa::simulation::PythonEnvironment::gil lock(__func__);
        
        default_excepthook = PySys_GetObject((char*)"excepthook") || fail("cannot get default excepthook");

        PyObject* self = PyCapsule_New(flags, NULL, NULL) || fail("cant wrap flags pointer");

        PyObject* excepthook = PyCFunction_NewEx(&except_hook_def, self, NULL)
                || fail("cannot create excepthook closure");

        PySys_SetObject((char*)"excepthook", excepthook) || fail("cannot set sys.excepthook");
    }
}


///////////////////////// Python_test Definition  //////////////////////////////////////////////////
Python_test::Python_test()
{
    static const std::string plugin = "SofaPython";
    sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);
}


void Python_test::run( const Python_test_data& data ) {

    msg_info("Python_test") << "running " << data.filepath;

    // Check the file exists
    std::ifstream file(data.filepath.c_str());
    bool scriptFound = file.good();
    if( !scriptFound )
        FAIL() << " missing file ("<<data.filepath<<")";

    char flags = 0;
    try {
        install_sys_excepthook(&flags);
    } catch( std::runtime_error& e) {
        FAIL() << "error setting up python excepthook, aborting test" << " ("<<data.filepath<<")";
    }

    try{
        if( !loader.loadTestWithArguments(data.filepath.c_str(),data.arguments) )
            FAIL() << " unable to run test ("<<data.filepath<<")";

        if( flags & flag::test_failure ) {
            FAIL() << "test failure in ("<<data.filepath<<")";
        }

        if( flags & flag::python_error) {
            FAIL() << "python error in ("<<data.filepath<<")";
        }
    } catch( simulation::PythonEnvironment::system_exit& e) {
        SUCCEED() << "test terminated normally";
    } catch( const result& test_result ) {
        ASSERT_TRUE(test_result.value) << " ("<<data.filepath<<")";
    }
}

static bool ends_with(const std::string& suffix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t ls = suffix.length();

    if(lf < ls) return false;

    return (0 == full.compare(lf - ls, ls, suffix));
}

static bool starts_with(const std::string& prefix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t lp = prefix.length();

    if(lf < lp) return false;

    return (0 == full.compare(0, lp, prefix));
}

void Python_test_list::addTestDir(const std::string& dir, const std::string& prefix) {

    std::vector<std::string> files;
    helper::system::FileSystem::listDirectory(dir, files);

    for(const std::string& file : files) {
        if( starts_with(prefix, file) && ends_with(".py", file) ) {
            addTest(file, dir);
        }
    }

}



///////////////////////// Python_scene_test Definition  ////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
struct Listener : core::objectmodel::BaseObject {

    Listener() {
        f_listening = true;
    }

    virtual void handleEvent(core::objectmodel::Event * event) {
        if (core::objectmodel::PythonScriptEvent::checkEventType(event)
                || core::objectmodel::ScriptEvent::checkEventType(event) )
        {
            core::objectmodel::ScriptEvent* e = static_cast<core::objectmodel::ScriptEvent*>(event);
            std::string name = e->getEventName();
            if( name == "success" ) {
                throw Python_scene_test::result(true);
            } else if (name == "failure") {
                throw Python_scene_test::result(false);
            }
        }
    }

};






Python_scene_test::Python_scene_test()
    : max_steps(1000) {

}

void Python_scene_test::run( const Python_test_data& data ) {

    msg_info("Python_scene_test") << "running "<< data.filepath;

    {
        // Check the file exists
        std::ifstream file(data.filepath.c_str());
        bool scriptFound = file.good();
        ASSERT_TRUE(scriptFound) << " ("<<data.filepath<<")";
    }

    if( !simulation::getSimulation() ) {
        simulation::setSimulation( new sofa::simulation::graph::DAGSimulation() );
    }

    char flags = 0;
    try {
        install_sys_excepthook(&flags);
    } catch( std::runtime_error& e) {
        ASSERT_TRUE(false) << "error setting up python excepthook, aborting test" << " ("<<data.filepath<<")";
    }

    simulation::Node::SPtr root;

    try {

        loader.loadSceneWithArguments(data.filepath.c_str(),
                                      data.arguments,
                                      &root);
        ASSERT_TRUE(bool(root)) << "scene creation failed!" << " ("<<data.filepath<<")";

        root->addObject( new Listener );
        simulation::getSimulation()->init(root.get());


        // TODO eventually tests should only stop by throwing SystemExit
        unsigned i;
        for(i = 0; (i < max_steps) && !(flags & flag::stop) && root->isActive(); ++i) {
            simulation::getSimulation()->animate(root.get(), root->getDt());
        }

        ASSERT_TRUE(i != max_steps) << "maximum allowed steps reached: " << max_steps << " ("<<data.filepath<<")";

        if( flags & flag::test_failure ) {
            FAIL() << "test failure";
        }

        if( flags & flag::python_error) {
            FAIL() << "python error";
        }

    } catch( simulation::PythonEnvironment::system_exit& e) {
        SUCCEED() << "test terminated normally";
    } catch( const result& test_result ) {
        ASSERT_TRUE(test_result.value) << " ("<<data.filepath<<")";

        // TODO raii for unloading
        simulation::getSimulation()->unload( root.get() );
    }
}



} // namespace sofa

