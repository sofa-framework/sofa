#ifndef PYTHONMODULE_SOFA_BINDING_BASE_H
#define PYTHONMODULE_SOFA_BINDING_BASE_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, boost::intrusive_ptr<Base>, true)

void moduleAddBase(py::module& m);

#endif /// PYTHONMODULE_SOFA_BINDING_BASE_H
