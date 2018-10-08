
#ifndef PYTHONMODULE_SOFA_BINDING_DATACONTAINER_H
#define PYTHONMODULE_SOFA_BINDING_DATACONTAINER_H

#include <pybind11/pybind11.h>
#include <sofa/core/objectmodel/BaseData.h>

namespace sofapython3
{
/// Makes an alias for the pybind11 namespace to increase readability.
namespace py { using namespace pybind11; }
using sofa::core::objectmodel::BaseData;

class scoped_write_access
{
public:
    BaseData* data{nullptr};
    void* ptr{nullptr};
    scoped_write_access(BaseData* data_) : data(data_){ ptr = data->beginEditVoidPtr(); }
    ~scoped_write_access(){ data->endEditVoidPtr(); }
};

class scoped_read_access
{
public:
    BaseData* data{nullptr};
    void* ptr{nullptr};
    scoped_read_access(BaseData* data_) : data(data_){ ptr = data->beginEditVoidPtr(); }
    ~scoped_read_access(){ data->endEditVoidPtr(); }
};

class scoped_writeonly_access
{
public:
    BaseData* data {nullptr};
    void* ptr{nullptr};
    scoped_writeonly_access(BaseData* data_) : data(data_){ ptr = data->beginEditVoidPtr(); }
    ~scoped_writeonly_access(){ data->endEditVoidPtr(); }
};

void moduleAddDataContainer(py::module& m);
void moduleAddDataAsString(py::module& m);
void moduleAddWriteAccessor(py::module& m);

class DataContainer : public BaseData {} ;
class DataAsString : public BaseData {} ;
}

#endif /// PYTHONMODULE_SOFA_BINDING_DATACONTAINER_H
