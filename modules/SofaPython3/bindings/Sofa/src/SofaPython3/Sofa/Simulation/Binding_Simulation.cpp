#include "Binding_Simulation.h"

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;
using sofa::simulation::Node;
using namespace pybind11::literals;

namespace sofapython3
{

void moduleAddSimulation(py::module &m)
{
  ///////////////////////////// Simulation binding //////////////////////////////
  py::class_<Simulation, Simulation::SPtr> s(m, "Simulation");
  s.def("print", &Simulation::print, "node"_a);
  s.def("init", &Simulation::init, "node"_a);
  s.def("animate", &Simulation::animate, "node"_a, "dt"_a = 0.0);
  s.def("reset", &Simulation::reset, "node"_a);
  s.def("load", &Simulation::load, "filename"_a);
  s.def("unload", &Simulation::unload, "node"_a);
}

void moduleAddRuntime(py::module &m)
{
  py::module singleRuntime = m.def_submodule("SingleSimulation");
  singleRuntime.def("setSimulation", [](Simulation *s){ sofa::simulation::setSimulation(s); });
  singleRuntime.def("getSimulation", [](){ return sofa::simulation::getSimulation(); });

  singleRuntime.def("print", [](Node* n){ sofa::simulation::getSimulation()->print(n); });
  singleRuntime.def("animate", [](Node* n, float dt=0.0){ sofa::simulation::getSimulation()->animate(n, dt); });
  singleRuntime.def("init", [](Node* n){ sofa::simulation::getSimulation()->init(n); });
  singleRuntime.def("reset", [](Node* n){ sofa::simulation::getSimulation()->reset(n); });
  singleRuntime.def("load", [](const std::string name){
       return sofa::simulation::getSimulation()->load(name.c_str());
  });
  singleRuntime.def("unload", [](Node* n){
      sofa::simulation::getSimulation()->unload(n);
  });

}


} /// namespace sofapython3
