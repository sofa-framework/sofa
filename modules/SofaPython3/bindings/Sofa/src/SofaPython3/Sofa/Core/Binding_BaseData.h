
#ifndef PYTHONMODULE_SOFA_BINDING_BASEDATA_H
#define PYTHONMODULE_SOFA_BINDING_BASEDATA_H

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

class scoped_write_access
{
public:
    BaseData *data;
    scoped_write_access(BaseData* data_) : data(data_){ data->beginEditVoidPtr(); }
    ~scoped_write_access(){ data->endEditVoidPtr(); }
};

class scoped_read_access
{
public:
    BaseData *data;
    scoped_read_access(BaseData* data_) : data(data_){ data->beginEditVoidPtr(); }
    ~scoped_read_access(){ data->endEditVoidPtr(); }
};

class scoped_writeonly_access
{
public:
    BaseData *data;
    scoped_writeonly_access(BaseData* data_) : data(data_){ data->beginEditVoidPtr(); }
    ~scoped_writeonly_access(){ data->endEditVoidPtr(); }
};


/// This class is used to mimick the smart pointer API needed by pybind to hold raw
/// pointer that are not released by python and are not using sharedptr (so weak_ptr
/// is not possible)
// TODO: move this somewhere else as we will probably need it in several other places.
template <class T> class raw_ptr
{
    public:
        raw_ptr() : ptr(nullptr) {}
        raw_ptr(T* ptr) : ptr(ptr) {}
        raw_ptr(const raw_ptr& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        void destroy() { delete ptr; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};

/// More info about smart pointer in
/// pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
/// BaseData are raw ptr so we use the raw_ptr smart pointer.
/// If you have a better way to do that, please make a PR.
PYBIND11_DECLARE_HOLDER_TYPE(BaseData, raw_ptr<BaseData>);

void moduleAddBaseData(py::module& m);
void moduleAddDataAsContainer(py::module& m);
void moduleAddDataAsString(py::module& m);
void moduleAddWriteAccessor(py::module& m);

class DataAsContainer : public BaseData {} ;
class DataAsString : public BaseData {} ;

#endif /// PYTHONMODULE_SOFA_BINDING_BASEDATA_H
