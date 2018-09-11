#include "Binding_BaseObject.h"

void moduleAddBaseObject(py::module& m)
{
    py::class_<BaseObject, Base, BaseObject::SPtr> p(m, "BaseObject");
}
