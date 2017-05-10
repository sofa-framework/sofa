#include <SofaPython/PythonScriptEvent.h>

#include "Python_test.h"

#include <sofa/helper/system/PluginManager.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>

#include <SofaPython/PythonMacros.h>
#include <SofaPython/PythonFactory.h>

namespace sofa {



Python_test::Python_test()
{
    static const std::string plugin = "SofaPython";
    sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);
}



void Python_test::run( const Python_test_data& data ) {

    msg_info("Python_test") << "running " << data.filepath;

    {
        // Check the file exists
        std::ifstream file(data.filepath.c_str());
        bool scriptFound = file.good();
        ASSERT_TRUE(scriptFound);
    }

    ASSERT_TRUE( loader.loadTestWithArguments(data.filepath.c_str(),data.arguments) );

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




static PyObject* except_hook(PyObject* self, PyObject* args) {
    PyObject* default_excepthook = nullptr;
    PyObject* py_run = nullptr;
    
    // parse upvalue
    PyArg_ParseTuple(self, "OO", &default_excepthook, &py_run);
    assert(default_excepthook); assert(py_run);

    // disable `run` flag
    using simulation::Node;
    bool* run = (bool*)PyCapsule_GetPointer(py_run, NULL);
    assert(run && "cannot get `run` flag");
    *run = false;
    
    // TODO we should probably decref std_except_hook/py_root/self at this point

    // TODO we should eventually distinguish between legit test failures
    // (e.g. catching assertion errors) vs. python errors
    
    // call standard excepthook
    return PyObject_CallObject(default_excepthook, args);
}

static PyMethodDef except_hook_def = {
    "sofa_excepthook",
    except_hook,
    METH_VARARGS,
    NULL
};


static void install_sys_excepthook(bool* run) {
    PyObject* sys = PyImport_ImportModule("sys") || fail("cannot import `sys` module");
    
    PyObject* sys_dict = PyModule_GetDict(sys) || fail("cannot import `sys` module dict");
    
    PyObject* default_excepthook = PyDict_GetItemString(sys_dict, "__excepthook__")
        || fail("cannot get default excepthook");

    PyObject* py_run = PyCapsule_New(run, NULL, NULL) || fail("cant wrap `run` flag");
    
    PyObject* self = PyTuple_Pack(2, default_excepthook, py_run) || fail("cannot pack `self`");
    
    PyObject* excepthook = PyCFunction_NewEx(&except_hook_def, self, NULL)
        || fail("cannot create excepthook closure");
    
    PyDict_SetItemString(sys_dict, "excepthook", excepthook) || fail("cannot set sys.excepthook");
}




void Python_scene_test::run( const Python_test_data& data ) {

    msg_info("Python_scene_test") << "running "<< data.filepath;

    {
        // Check the file exists
        std::ifstream file(data.filepath.c_str());
        bool scriptFound = file.good();
        ASSERT_TRUE(scriptFound);
    }

    if( !simulation::getSimulation() ) {
        simulation::setSimulation( new sofa::simulation::graph::DAGSimulation() );
    }

    bool run = true;
    try {
        install_sys_excepthook(&run);
    } catch( std::runtime_error& e) {
        ASSERT_TRUE(false) << "error setting up python excepthook, aborting test";
    }
    
    simulation::Node::SPtr root;
    loader.loadSceneWithArguments(data.filepath.c_str(),
                                  data.arguments,
                                  &root);
    ASSERT_TRUE(bool(root)) << "scene creation failed!";

    root->addObject( new Listener );
	simulation::getSimulation()->init(root.get());

	try {
		while(run && root->isActive()) {
			simulation::getSimulation()->animate(root.get(), root->getDt());
		}
        ASSERT_TRUE(run) << "python error occurred";
	} catch( const result& test_result ) {
        ASSERT_TRUE(test_result.value);
        simulation::getSimulation()->unload( root.get() );
	}
}



} // namespace sofa

