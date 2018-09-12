#include "Binding_Simulation.h"

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;


void moduleAddSimulation(py::module &m)
{
  py::class_<Simulation, Simulation::SPtr> s(m, "Simulation");
  s.def("print", &Simulation::print, "root"_a);
  s.def("init", &Simulation::init, "root"_a);
  s.def("animate", &Simulation::animate, "root"_a, "dt"_a = 0.0);
  s.def("reset", &Simulation::reset, "root"_a);
  s.def("load", &Simulation::load, "filename"_a);
  s.def("unload", &Simulation::unload, "root"_a);
}
