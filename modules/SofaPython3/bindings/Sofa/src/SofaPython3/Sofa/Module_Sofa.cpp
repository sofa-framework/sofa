#include "Core/Submodule_Core.h"
#include "Simulation/Submodule_Simulation.h"

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{
    py::module core = addSubmoduleCore( m );
    py::module simulation = addSubmoduleSimulation( m );

    /// Import into the Sofa main package the class from theyr sub-module.
    m.add_object("PythonController", core.attr("PythonController"));
    m.add_object("Node", simulation.attr("Node"));
}
