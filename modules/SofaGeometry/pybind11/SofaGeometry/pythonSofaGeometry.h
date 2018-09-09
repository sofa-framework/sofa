#ifndef PYTHONSOFAGEOMETRY_H
#define PYTHONSOFAGEOMETRY_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

void init_vec3(py::module &);
void init_ray(py::module &);
void init_plane(py::module &);

#endif  // PYTHONSOFAGEOMETRY_H
