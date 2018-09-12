#include "Binding_Node.h"

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

void moduleAddNode(py::module &m)
{
  py::class_<Node, Base, Node::SPtr> p(m, "Node");

  // TODO why is calling base function still crash with a segfault ?
  // Si je ne met pas la ligne commenté suivante
  p.def("getName", [](Node& self){
      return self.getName();
  });
  // Ca crash quand ça appelle getName()
  // Je suppose que c'est lié à un cast qui fait un smart:ptr null
  // Ca fait pareil au niveau de getData...bref tout ce qui est hérité.

  // TODO make an factory helper mecanisme
  p.def("createObject", [](Node& self, const std::string& s){
      py::print("createObject");
  });

  p.def("createChild", &Node::createChild);
}
