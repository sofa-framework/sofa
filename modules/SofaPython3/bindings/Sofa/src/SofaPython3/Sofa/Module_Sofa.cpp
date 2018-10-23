#include "Core/Submodule_Core.h"
#include "Helper/Submodule_Helper.h"
#include "Simulation/Submodule_Simulation.h"
#include "Types/Submodule_Types.h"

namespace sofapython3
{

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{    
    py::module core = addSubmoduleCore(m);
    py::module simulation = addSubmoduleSimulation(m);
    py::module helper = addSubmoduleHelper(m);

    /// Import into the Sofa main package the class from theyr sub-module.
    m.add_object("PythonController", core.attr("PythonController"));
    m.add_object("ForceField", core.attr("ForceField"));
    m.add_object("Node", simulation.attr("Node"));

    m.add_object("msg_info", helper.attr("msg_info"));
    m.add_object("msg_warning", helper.attr("msg_warning"));
    m.add_object("msg_error", helper.attr("msg_error"));
    m.add_object("msg_fatal", helper.attr("msg_fatal"));
    m.add_object("msg_deprecated", helper.attr("msg_deprecated"));

    /// Import some binding part fully written in python (because I'm too lazy)
    py::module st = m.def_submodule("Types");
    py::module stimpl = py::module::import("__Sofa_Types__");
    st.add_object("RGBAColor", stimpl.attr("RGBAColor"));
    st.add_object("Vec3", stimpl.attr("Vec3"));

}

} ///namespace sofapython3
