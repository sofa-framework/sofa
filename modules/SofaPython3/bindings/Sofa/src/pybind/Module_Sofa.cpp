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

    m.def("test", [](){
        py::module m = py::module::import("SofaRuntime");
        std::cout << "Hello..." << std::endl;
        Node::SPtr s=Node::create("Damien");
        std::cout << "ZUT  " << (void*)s.get() << std::endl;
        std::cout << "ZOU  " << s->getName() << std::endl;
        Base::SPtr b=Base::SPtr(s);
        return s; //py::cast(b);
    });
}
