#include "Binding_BaseObject.h"
#include "Binding_PythonController.h"

void moduleAddBaseObject(py::module& m)
{
    py::class_<BaseObject, Base, BaseObject::SPtr>p(m, "BaseObject");
    p.def("init", &BaseObject::init);
    p.def("reinit", &BaseObject::init);

    p.def("getPathName", &BaseObject::getPathName);
    p.def("getLinkPath", [](const BaseObject &self)
    {
        return std::string("@") + self.getPathName();
    });
}
