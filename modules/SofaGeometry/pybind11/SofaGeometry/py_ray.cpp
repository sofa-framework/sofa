#include "SofaGeometry/Ray.h"
#include "pythonSofaGeometry.h"

using sofageometry::Ray;
typedef sofageometry::Vec3d vec3;

void init_ray(py::module &m) {
  py::class_<Ray> r(m, "Ray");
  r.def(py::init<>());
  r.def(py::init<const vec3 &, const vec3 &>(), "origin"_a, "direction"_a);
  r.def("getPoint", &Ray::getPoint, "distance"_a);
  r.def_readwrite("direction", &Ray::direction);
  r.def_readwrite("origin", &Ray::origin);
}
