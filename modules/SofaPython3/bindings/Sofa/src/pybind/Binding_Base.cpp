#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseLink.h>
using sofa::core::objectmodel::BaseLink;

#include "Binding_BaseData.h"

void moduleAddBase(py::module &m)
{
    py::class_<Base, Base::SPtr> p(m, "Base");
    p.def_property("name", &Base::getName,
                   [](Base &self, const std::string &s) { self.setName(s); });

    p.def("getData", [](Base& self, const std::string& s) -> py::object
    {
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            if(d->getValueTypeInfo()->Container())
                return py::cast(reinterpret_cast<BaseDataAsContainer*>(d));
            return py::cast(d);
        }
        return py::none();
    });

    p.def("__getattr__", [](Base& self, const std::string& s) -> py::object
    {
        /// I'm not sure implicit behavior is nice but we could do:
        ///    - The attribute is a data,
        ///         returns it if it is a container
        ///         returns the value/specific binding otherwise
        ///    - The attribute is a link, return it.
        ///    - The attribute is an object or a child return it.
        ///    - The attribute is not existing:
        ///                raise an exception or search using difflib for close match.

        std::cout << "(__getattr__ to implement)" << std::endl ;
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            return py::cast(d);
        }
        return py::none();
    });

    p.def("__setattr__", [](Base& self, const std::string& s, py::object& value) -> py::object
    {
        /// I'm not sure implicit behavior is nice but we could do:
        ///    - The attribute is a data, set its value.
        ///          If the data is a container...check dimmensions and do type coercion.
        ///    - The attribute is a link, set its value.
        ///    - The attribute is an object or a child, raise an exception.
        ///    - The attribute is not existing, add it has data with type deduced from value ?
        std::cout << "(__setattr__ to implement)" << std::endl ;
        BaseData* d = self.findData(s);
        if(d!=nullptr)
            return py::cast(d);

        BaseLink* l = self.findLink(s);
        if(l!=nullptr)
            return py::cast(l);

        /// To implement the remaining we needs to have a context...so we need to override the
        /// binding in: BaseObject & Node
        return py::none();
    });

}
