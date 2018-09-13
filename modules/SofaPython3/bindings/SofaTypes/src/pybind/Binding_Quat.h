#ifndef PYTHONMODULE_SOFA_BINDING_QUAT_H
#define PYTHONMODULE_SOFA_BINDING_QUAT_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/defaulttype/Quat.h>
using sofa::defaulttype::Quat;

void moduleAddQuat(py::module& m);

#endif  // PYTHONMODULE_SOFA_BINDING_QUAT_H
