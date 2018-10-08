#include <pybind11/numpy.h>
#include <pybind11/eval.h>

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseObject.h>
using  sofa::core::objectmodel::BaseObject;

#include <sofa/core/objectmodel/BaseNode.h>
using  sofa::core::objectmodel::BaseNode;

#include "Binding_Base.h"
#include "Binding_BaseData.h"
#include "DataHelper.h"

namespace sofapython3
{

void moduleAddBaseData(py::module& m)
{
    py::class_<BaseData, raw_ptr<BaseData>> p(m, "Data");
    p.def("setName", &BaseData::setName);
    p.def("getName", &BaseData::getName);
    p.def("getCounter", [](BaseData& d){ return d.getCounter(); } );
    p.def("getHelp", &BaseData::getHelp);
    p.def("unset", &BaseData::unset);
    p.def("getOwner", &BaseData::getOwner);

    // TODO: Implementation should look like: https://github.com/sofa-framework/sofa/issues/767
    p.def("__setitem__", [](BaseData& self, py::object& key, py::object& value)
    {
        std::cout << "mapping protocol, __setitem__ to implement)" << std::endl ;
        return py::none();
    });

    p.def("__len__", [](BaseData& b) -> size_t
    {
        auto nfo = b.getValueTypeInfo();
        return nfo->size(b.getValueVoidPtr()) / nfo->size();
    });

    p.def("getPath", [](BaseData& self)
    {
        Base* b= self.getOwner();
        std::string prefix = getPathTo(b);
        return prefix+"."+self.getName();
    });

    p.def("__str__", [](BaseData* self)
    {
        return py::str(toPython(self));
    });

    p.def("__repr__", [](BaseData* self)
    {
        return py::repr(toPython(self));
    });
}

} /// namespace sofapython3
