#include "Binding_Simulation.h"

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;
using namespace pybind11::literals;

void moduleAddSimulation(py::module &m)
{
  py::class_<Simulation, Simulation::SPtr> s(m, "Simulation");
  s.def("print", &Simulation::print, "node"_a);
  s.def("init", &Simulation::init, "node"_a);
  s.def("animate", &Simulation::animate, "node"_a, "dt"_a = 0.0);
  s.def("reset", &Simulation::reset, "node"_a);
  s.def("load", &Simulation::load, "filename"_a);
  s.def("unload", &Simulation::unload, "node"_a);
}
