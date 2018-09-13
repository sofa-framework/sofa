#ifndef PYTHONMODULE_SOFA_BINDING_COLOR_H
#define PYTHONMODULE_SOFA_BINDING_COLOR_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/defaulttype/Color.h>
using sofa::defaulttype::RGBAColor;

void moduleAddColor(py::module& m);

#endif  // PYTHONMODULE_SOFA_BINDING_COLOR_H
