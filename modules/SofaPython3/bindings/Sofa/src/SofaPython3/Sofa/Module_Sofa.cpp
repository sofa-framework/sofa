#include "Core/Submodule_Core.h"
#include "Simulation/Submodule_Simulation.h"

namespace sofapython3
{

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{    
    py::module core = addSubmoduleCore( m );
    py::module simulation = addSubmoduleSimulation( m );

    /// Import into the Sofa main package the class from theyr sub-module.
    m.add_object("PythonController", core.attr("PythonController"));
    m.add_object("ForceField", core.attr("ForceField"));
    m.add_object("Node", simulation.attr("Node"));

    /// Import some binding part fully written in python (because I'm too lazy)
    //py::module types = m.def_submodule("Types");
    //py::module pytypes = py::module::import("Types");
    //types.add_object("vec3", pytypes.attr("vec3"));
    //types.add_object("quat", pytypes.attr("quat"));
}

} ///namespace sofapython3
