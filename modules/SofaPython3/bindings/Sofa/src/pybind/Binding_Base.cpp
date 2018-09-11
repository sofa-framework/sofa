#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, boost::intrusive_ptr<Base>, true)

void init_Base(py::module &m)
{
  py::class_<Base, Base::SPtr> p(m, "Base");
  p.def("setName", [](Base& self, const std::string& s){ self.setName(s); });
  p.def("getName", &Base::getName);
}


#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

void init_BaseObject(py::module& m)
{
    py::class_<BaseObject, Base, BaseObject::SPtr> p(m, "BaseObject");
}

class  PythonController : public BaseObject
{
public:
    SOFA_CLASS(PythonController, BaseObject);

    PythonController(){}
};

void init_PythonController(py::module& m)
{
    py::class_<PythonController, BaseObject, PythonController::SPtr> p(m, "PythonController");
}

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;

void init_Node(py::module &m)
{
  py::class_<Node, Base, Node::SPtr> p(m, "Node");
  p.def("createObject", [](Node& self, const std::string& s){
      py::print("createObject");
  });

  p.def("createChild", [](Node& self, const std::string& s) -> Node::SPtr {
      py::print("createChild");
      return Node::SPtr(&self);
  });
}

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{
    init_Base(m);
    init_BaseObject(m);
    init_Node(m);
}
