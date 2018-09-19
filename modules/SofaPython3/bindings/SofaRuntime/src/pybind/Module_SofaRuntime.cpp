#include <pybind11/eval.h>
namespace py = pybind11;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi = sofa::simpleapi;

#include <sofa/helper/Utils.h>
using sofa::helper::Utils;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::PluginRepository;

/// here is the .pxd.
/// TODO(bruno-marques) ... this is mandatory for the conversion. But the path must be made
/// un-ambiguous to sofa.
/// eg:
/// #include <PythonModule_Sofa/Binding_Node>
/// #include <Sofa/python/Binding_Node>
#include <src/pybind/Binding_Node.h>
#include <src/pybind/Binding_Base.h>

// TODO (je suis dans une WIP) donc je peux dire que c'est beurk
#include <SofaSimulationCommon/init.h>
#include <SofaSimulationGraph/init.h>

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(SofaRuntime, m) {
    // TODO, ces trucs sont fort laid. Normalement ce devrait être une joli plugin qui
    // appelle le init.
    sofa::simulation::common::init();
    sofa::simulation::graph::init();
    //    sofa::simulation::setSimulation(new DAGSimulation());

    // Add the plugin directory to PluginRepository
    const std::string& pluginDir = Utils::getPluginDirectory();
    PluginRepository.addFirstPath(pluginDir);

    /// We need to import the project dependencies
    py::module::import("Sofa");

    m.def("getSimulation", [](){
        return sofa::simulation::getSimulation(); });
    
    m.def("importPlugin", [](const std::string& name)
    {
        return simpleapi::importPlugin(name);
    });

    m.def("reinit", []()
    {
        /// set the Simulation, replacing the existing one (which is automatically deleted)
        if( !sofa::simulation::getSimulation() )
            sofa::simulation::setSimulation(new DAGSimulation());
    });


    /// py::module runtime = m.def_submodule("Runtime");
    /// runtime.add_object();
    /// py::exec("import SofaRuntime as Runtime", py::globals());

    m.def("dev_getANode", []() {
        Node::SPtr n = Node::create("testNode");
        return n;
    });

    m.def("loadScene", [](const std::string& filename) -> py::object
    {
        /// set the Simulation, replacing the existing one (which is automatically deleted)
        if( !sofa::simulation::getSimulation() )
            sofa::simulation::setSimulation(new DAGSimulation());

        /// Evaluate the content of the file in the scope of the main module
        py::object globals = py::module::import("__main__").attr("__dict__");
        py::object locals = py::dict();
        py::eval_file(filename, globals, locals);

        if( locals.contains("createScene") ){
            py::object o = locals["createScene"];
            if( py::isinstance<py::function>(o) ){
                Node::SPtr tmp = Node::create("root");
                std::cout << "ICI" << std::endl;
                //PAUSE-WORK-HERE
                // Je me suis arrête ici. Ca compile mais ça ne marche pas.
                // quand je charge le module:
                //    import sys
                //    sys.path.append("./SofaRuntime/package")
                //    import SofaRuntime
                //    SofaRuntime.loadScene("Sofa/examples/example1.py")
                // Ca affiche:
                // Traceback (most recent call last):
                //    File "<stdin>", line 1, in <module>
                //    RuntimeError: make_tuple(): unable to convert arguments to Python object (compile in debug mode for details)

                o(tmp);
                std::cout << "LA: " << tmp->getName() << std::endl;
                return py::cast(tmp);
            }
        }
        return py::none();
    });
}
