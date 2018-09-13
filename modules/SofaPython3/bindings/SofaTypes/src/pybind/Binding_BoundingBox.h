#ifndef PYTHONMODULE_SOFA_BINDING_BOUNDINGBOX_H
#define PYTHONMODULE_SOFA_BINDING_BOUNDINGBOX_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/defaulttype/BoundingBox.h>
using sofa::defaulttype::BoundingBox;
using sofa::defaulttype::BoundingBox1D;
using sofa::defaulttype::BoundingBox2D;

void moduleAddBoundingBox(py::module& m);

#endif  // PYTHONMODULE_SOFA_BINDING_BOUNDINGBOX_H
