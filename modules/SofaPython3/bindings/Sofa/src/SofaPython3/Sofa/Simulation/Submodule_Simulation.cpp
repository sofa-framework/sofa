#include "Submodule_Simulation.h"

#include "Binding_Node.h"
#include "Binding_Simulation.h"

namespace sofapython3
{
/// The first parameter must be named the same as the module file to load.
py::module addSubmoduleSimulation(py::module& module)
{
  py::module simu = module.def_submodule("Simulation");
  moduleAddNode(simu);
  moduleAddSimulation(simu);

  py::module runtime = module.def_submodule("Runtime");
  moduleAddRuntime(simu);

  return simu;
}

} ///namespace sofapython3
