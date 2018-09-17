#ifndef PYTHONMODULE_SOFA_BINDING_MAT_H
#define PYTHONMODULE_SOFA_BINDING_MAT_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;


#include <sofa/defaulttype/Mat.h>
using sofa::defaulttype::Mat;
#include "Binding_Vec.h"

namespace pyMat {
template <int R, int C>
std::string __str__(const Mat<R, C, double> &self, bool repr = false);
} // namespace pyMat

void moduleAddMat(py::module& m);

#endif  // PYTHONMODULE_SOFA_BINDING_MAT_H
