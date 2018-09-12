#include "Binding_Node.h"

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

void moduleAddNode(py::module &m)
{
  py::class_<Node, Base, Node::SPtr> p(m, "Node");

  //JE ME SUIS ARRETE ICI
  // Si je ne met pas la ligne commenté suivante
  p.def("getName", [](Node& self){
      return self.getName();
  });

  /*p.def("getData", [](Node& self, const std::string& s) -> py::object
  {
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            return py::cast(d);
        }
        return py::none();
  });*/

  // Ca crash quand ça appelle getName()
  // Je suppose que c'est lié à un cast qui fait un smart:ptr null
  // Ca fait pareil au niveau de getData...


  p.def("createObject", [](Node& self, const std::string& s){
      py::print("createObject");
  });

  p.def("createChild", [](Node& self, const std::string& s) -> py::object {
      py::print("createChild");
      return py::cast(&self);
  });
}
