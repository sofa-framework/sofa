#include "Binding_Node.h"

#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi = sofa::simpleapi ;

void moduleAddNode(py::module &m)
{
  py::class_<Node, Base, Node::SPtr> p(m, "Node");

  // // TODO why is calling base function still crash with a segfault ?
  // // Si je ne met pas la ligne commenté suivante
  // p.def("getName", [](Node& self){
  //     return self.getName();
  // });
  // // Ca crash quand ça appelle getName()
  // // Je suppose que c'est lié à un cast qui fait un smart:ptr null
  // // Ca fait pareil au niveau de getData...bref tout ce qui est hérité.

  // TODO make an factory helper mecanisme
  p.def("createObject", [](Node& self, const std::string& type){
    std::cout << "CREATE OBJECT" << std::endl;
      return py::cast( simpleapi::createObject(&self, type) );
  });

  p.def("createChild", &Node::createChild);
  p.def("getRoot", &Node::getRoot);

  // TODO in Sofa.Runtime ?
  p.def("simulationStep", [](Node &n, double dt) {
      sofa::simulation::getSimulation()->animate(&n, dt);
      }, "dt"_a);

  p.def("reset", [](Node &n) { sofa::simulation::getSimulation()->reset(&n); });
  p.def("init", [](Node &n) { sofa::simulation::getSimulation()->init(&n); });

  p.def("getChild", [](Node &n, const std::string &name) -> py::object {
    sofa::simulation::Node *child = n.getChild(name);
    if (child)
      return py::cast(child);
    else
      return py::none();
  });
}
