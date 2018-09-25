#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseLink.h>
using sofa::core::objectmodel::BaseLink;

#include "Binding_BaseData.h"
using sofa::defaulttype::AbstractTypeInfo;

py::object toPython(BaseData* d)
{
    const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };
    if(nfo.Container())
    {
        size_t dim0 = nfo.size(d->getValueVoidPtr())/nfo.size();
        size_t dim1 = nfo.size();
        py::list list;
        for(size_t i=0;i<dim0;i++)
        {
            py::list list1;
            for(size_t j=0;j<dim1;j++)
            {
                if(nfo.Integer())
                    list1.append(nfo.getIntegerValue(d->getValueVoidPtr(),i*dim1+j));
                if(nfo.Scalar())
                    list1.append(nfo.getScalarValue(d->getValueVoidPtr(),i*dim1+j));
                if(nfo.Text())
                    list1.append(nfo.getTextValue(d->getValueVoidPtr(),0));
            }
            list.append(list1);
        }
        return list;
    }

    if(nfo.Integer())
        return py::cast(nfo.getIntegerValue(d->getValueVoidPtr(), 0));
    if(nfo.Text())
        return py::cast(d->getValueString());
    if(nfo.Scalar())
        return py::cast(nfo.getScalarValue(d->getValueVoidPtr(), 0));

    return py::cast(d->getValueString());
}

void fromPython(BaseData* d, py::object& o)
{
    const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

    if(nfo.Integer())
        nfo.setIntegerValue(d->beginEditVoidPtr(), 0, py::cast<int>(o));
    if(nfo.Text())
        nfo.setTextValue(d->beginEditVoidPtr(), 0, py::cast<py::str>(o));
    if(nfo.Scalar())
        nfo.setScalarValue(d->beginEditVoidPtr(), 0, py::cast<double>(o));
    d->endEditVoidPtr();
    msg_error("SofaPython3") << "binding problem";
}

py::object BindingBase::GetAttr(Base& self, const std::string& s)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data,
    ///         returns it if it is a container
    ///         returns the value/specific binding otherwise
    ///    - The attribute is a link, return it.
    ///    - The attribute is an object or a child return it.
    ///    - The attribute is not existing:
    ///                raise an exception or search using difflib for close match.
    BaseData* d = self.findData(s);
    if(d!=nullptr)
    {
        const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

        if(nfo.Container())
            return py::cast(reinterpret_cast<DataAsContainer*>(d));
        if(nfo.Text())
            return py::cast(reinterpret_cast<DataAsString*>(d));
        return toPython(d);
    }

    throw py::attribute_error(s);
}

void BindingBase::SetAttr(py::object self, const std::string& s, py::object& value)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data, set its value.
    ///          If the data is a container...check dimmensions and do type coercion.
    ///    - The attribute is a link, set its value.
    ///    - The attribute is an object or a child, raise an exception.
    ///    - The attribute is not existing, add it has data with type deduced from value ?
    Base& self_base = py::cast<Base&>(self);
    BaseData* d = self_base.findData(s);

    if(d!=nullptr)
    {
        const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

        /// We go for the container path.
        if(nfo.Container())
        {
            return;
        }
        fromPython(d, value);
        return;
    }

    BaseLink* l = self_base.findLink(s);
    if(l!=nullptr)
    {
        return;
    }

    /// We are falling back to dynamically adding the objet into the object dict.
    py::dict t = self.attr("__dict__");
    if(!t.is_none())
    {
        t[s.c_str()] = value;
        return;
    }

    /// Well this should never happen unless there is no __dict__
    throw py::attribute_error();
}

void moduleAddBase(py::module &m)
{
    py::class_<Base, Base::SPtr> p(m, "Base");
    p.def("getData", [](Base& self, const std::string& s) -> py::object
    {
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

            if(nfo.Container())
                return py::cast(reinterpret_cast<DataAsContainer*>(d));
            if(nfo.Text())
                return py::cast(reinterpret_cast<DataAsString*>(d));
            return py::cast(d);
        }
        return py::none();
    });


    p.def("__getattr__", &BindingBase::GetAttr);
    p.def("__setattr__", &BindingBase::SetAttr);
}
