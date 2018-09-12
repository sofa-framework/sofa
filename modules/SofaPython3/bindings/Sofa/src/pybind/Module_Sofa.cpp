#include "Binding_Base.h"
#include "Binding_BaseData.h"
#include "Binding_BaseObject.h"
#include "Binding_Node.h"

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{
    moduleAddBase(m);
    moduleAddBaseData(m);
    moduleAddBaseObject(m);
    moduleAddNode(m);

    m.def("test", []()
    {
        py::module m = py::module::import("SofaRuntime");
        Node::SPtr n=Node::create("TestNode");
        Base::SPtr b=Base::SPtr(n);
        py::tuple t {2};                  /// Why there is no initializer list ?
        t[0] = b;
        t[1] = n;
        return t;
    });
}
