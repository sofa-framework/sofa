#ifndef PYTHONMODULE_SOFA_BINDING_VEC_H
#define PYTHONMODULE_SOFA_BINDING_VEC_H

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec;

namespace pyVec {
template <int N, class T>
std::string __str__(const Vec<N, T> &self, bool repr = false);
} // namespace pyVec

void moduleAddVec(py::module &m);

#endif // PYTHONMODULE_SOFA_BINDING_VEC_H
