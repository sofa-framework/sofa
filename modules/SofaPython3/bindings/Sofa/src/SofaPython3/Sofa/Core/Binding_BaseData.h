
#ifndef PYTHONMODULE_SOFA_BINDING_BASEDATA_H
#define PYTHONMODULE_SOFA_BINDING_BASEDATA_H

#include <sofa/core/objectmodel/BaseData.h>
#include "DataHelper.h"

/// More info about smart pointer in
/// pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
/// BaseData are raw ptr so we use the raw_ptr smart pointer.
/// If you have a better way to do that, please make a PR.
PYBIND11_DECLARE_HOLDER_TYPE(BaseData, sofapython3::raw_ptr<BaseData>)

namespace sofapython3
{
    void moduleAddBaseData(py::module& m);
} /// sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_BASEDATA_H
