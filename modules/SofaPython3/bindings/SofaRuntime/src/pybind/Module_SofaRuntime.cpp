#include <pybind11/eval.h>
namespace py = pybind11;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

/// here is the .pxd.
/// TODO(bruno-marques) ... this is mandatory for the conversion. But the path must be made
/// un-ambiguous to sofa.
/// eg:
/// #include <PythonModule_Sofa/Binding_Node>
/// #include <Sofa/python/Binding_Node>
#include <src/pybind/Binding_Node.h>

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(SofaRuntime, m) {
    /// We need to import the project dependencies
    py::module::import("Sofa");
    m.def("reinit", []()
    {
        if(sofa::simulation::getSimulation())
            delete sofa::simulation::getSimulation();
        sofa::simulation::setSimulation(new DAGSimulation());
    });

    m.def("loadScene", [](const std::string& filename) -> py::object {
        //if(sofa::simulation::getSimulation())
        //    delete sofa::simulation::getSimulation();
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
