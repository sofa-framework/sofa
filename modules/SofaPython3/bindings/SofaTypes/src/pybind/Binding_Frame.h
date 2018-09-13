#ifndef PYTHONMODULE_SOFA_BINDING_FRAME_H
#define PYTHONMODULE_SOFA_BINDING_FRAME_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/defaulttype/Frame.h>
using sofa::defaulttype::Frame;

void moduleAddFrame(py::module& m);

#endif  // PYTHONMODULE_SOFA_BINDING_FRAME_H
