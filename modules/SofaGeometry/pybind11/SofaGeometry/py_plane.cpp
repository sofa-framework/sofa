#include "SofaGeometry/Plane.h"
#include "SofaGeometry/Ray.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;
using sofageometry::Constants;
using sofageometry::Plane;
using sofageometry::Ray;
typedef sofageometry::Vec3d vec3;

void init_plane(py::module &m) {
  py::class_<Plane> p(m, "Plane");
  p.def_readwrite("distance", &Plane::distance);
  p.def_readwrite("normal", &Plane::normal);
  p.def(py::init<const vec3 &, double>(), "normal"_a = Constants::XAxis,
        "distance"_a = 0);

  p.def(py::init([](py::list l, double dist) {
          return std::unique_ptr<Plane>(
              new Plane(vec3(double(l[0].cast<py::float_>()),
                             double(l[1].cast<py::float_>()),
                             double(l[2].cast<py::float_>())),
                        dist));
        }),
        "normal"_a = Constants::XAxis, "distance"_a = 0);

  p.def(py::init<const vec3 &, const vec3 &>(), "normal"_a, "point"_a);
  p.def("raycast",
        (bool (Plane::*)(const Ray &, double &) const) & Plane::raycast,
        "ray"_a, "p"_a);
  p.def("raycast",
        [](const Plane &plane, Ray r) {
          double p = 0.0;
          plane.raycast(r, p);
          return p;
        },
        "ray"_a);
}
