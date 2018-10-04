#ifndef PYTHONMODULE_SOFA_BINDING_BASE_H
#define PYTHONMODULE_SOFA_BINDING_BASE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, sofa::core::sptr<Base>, true)

class BindingBase
{
public:
    static py::object GetAttr(Base& self, const std::string& s);
    static void SetAttr(py::object self, const std::string& s, py::object& value);
    static void SetAttrFromArray(py::object self, const std::string& s, const pybind11::array &value);
};

py::buffer_info toBufferInfo(BaseData& m);
bool hasArrayFor(BaseData* d);
py::array resetArrayFor(BaseData* d);
py::array getPythonArrayFor(BaseData* d);
py::object convertToPython(BaseData* d);

class DataDict
{
public:
    Base::SPtr owner;
    DataDict(Base::SPtr b){ owner = b; }
};

class DataDictIterator
{
public:
    Base::SPtr owner;
    size_t     index=0;
    bool       key;
    bool       value;
    DataDictIterator(Base::SPtr owner_, bool withKey, bool withValue)
    {
        owner=owner_;
        index=0;
        key=withKey;
        value=withValue;
    }
};

py::object toPython(BaseData* d, bool writeable=false);
void fromPython(BaseData* d, const pybind11::object &o);

void moduleAddDataDict(py::module& m);
void moduleAddDataDictIterator(py::module& m);
void moduleAddBase(py::module& m);

#endif /// PYTHONMODULE_SOFA_BINDING_BASE_H
