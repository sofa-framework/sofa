#ifndef PYTHONMODULE_SOFA_BINDING_BASE_H
#define PYTHONMODULE_SOFA_BINDING_BASE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, sofa::core::sptr<Base>, true)

template class pybind11::class_<sofa::core::objectmodel::Base,
                                sofa::core::objectmodel::Base::SPtr>;

namespace sofapython3
{
/// Makes an alias for the pybind11 namespace to increase readability.
namespace py { using namespace pybind11; }
using namespace pybind11::literals;
using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseData;
using sofa::core::sptr;

class BindingBase
{
public:
    static py::object GetAttr(Base* self, const std::string& s, bool doThrowException=true);
    static void SetAttr(py::object self, const std::string& s, pybind11::object value, bool withDict=false);
    static void SetAttr(Base& self, const std::string& s, py::object value);
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
    sptr<Base> owner;
    DataDict(sptr<Base> b){ owner = b; }
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

void moduleAddDataDict(py::module& m);
void moduleAddDataDictIterator(py::module& m);
void moduleAddBase(py::module& m);



} /// namespace sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_BASE_H
