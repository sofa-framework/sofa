#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseData.h"
#include "Binding_Simulation.h"
#include "Binding_Node.h"

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m) {
  moduleAddBase(m);
  moduleAddBaseData(m);
  moduleAddBaseObject(m);
  moduleAddSimulation(m);
  moduleAddNode(m);

  /// py::module runtime = m.def_submodule("Runtime");
  /// runtime.add_object();
  /// py::exec("import SofaRuntime as Runtime", py::globals());

  m.def("test", []() {
    py::module m = py::module::import("SofaRuntime");
    Node::SPtr n = Node::create("testNode");
    Base::SPtr b = Base::SPtr(n);
    py::tuple t{2}; /// Why there is no initializer list ?
    t[0] = b;
    t[1] = n;
    return t;
  });
}
