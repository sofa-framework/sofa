#include "pythonSofaGeometry.h"
#include <SofaGeometry/Constants.h>

typedef sofageometry::Vec3d vec3;

void init_vec3(py::module &m) {
  py::class_<vec3> v(m, "Vec3");
  v.def(py::init<double, double, double>());
  v.def(py::init<const vec3 &>());
  v.def(py::init([](py::list l) {
    return std::unique_ptr<vec3>(
        new vec3(double(l[0].cast<py::float_>()),
                 double(l[1].cast<py::float_>()),
                 double(l[2].cast<py::float_>())));
  }));
  v.def(py::init<>());

  v.def("set", &vec3::set<3>, "x"_a, "y"_a, "z"_a);

  v.def_property("x", [](vec3 &v) { return v.x(); },
                 [](vec3 &v, double x) { v.x() = x; })
      .def_property("y", [](vec3 &v) { return v.y(); },
                    [](vec3 &v, double y) { v.y() = y; })
      .def_property("z", [](vec3 &v) { return v.z(); },
                    [](vec3 &v, double z) { v.z() = z; })

      .def_property("xy",
                    [](vec3 &v) {
                      py::tuple t(2);
                      t[0] = v.x();
                      t[1] = v.y();
                      return t;
                    },
                    [](vec3 &v, double x, double y) {
                      v.x() = x;
                      v.y() = y;
                    })
      .def_property("xyz",
                    [](vec3 &v) {
                      py::tuple t(3);
                      t[0] = v.x();
                      t[1] = v.y();
                      t[2] = v.z();
                      return t;
                    },
                    [](vec3 &v, double x, double y, double z) {
                      v.x() = x;
                      v.y() = y;
                      v.z() = z;
                    });

  v.def(py::self != py::self)
      .def(py::self * py::self)
      .def(py::self * float())
      .def(py::self *= float())
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self);

  v.def("__getitem__",
        [](const vec3 &v, size_t i) {
          if (i >= v.size())
            throw py::index_error();
          return v[i];
        })
      .def("__setitem__", [](vec3 &v, size_t i, double d) {
        if (i >= v.size())
          throw py::index_error();
        double &val = v[i];
        val = d;
        return val;
      });

  v.def("normalize", [](vec3 &v) { return v.normalize(); })
      .def("normalized", &vec3::normalized)
      .def("norm", &vec3::norm)
      .def("norm2", &vec3::norm2)
      .def("divscalar", &vec3::divscalar<double>, "f"_a)
      .def("mulscalar", &vec3::mulscalar<double>, "f"_a)
      .def("eqmulscalar", &vec3::eqmulscalar<double>, "f"_a)
      .def("eqdivscalar", &vec3::eqdivscalar<double>, "f"_a)
      .def("lineardiv", &vec3::linearDivision<double>, "v"_a)
      .def("linearmul", &vec3::linearProduct<double>, "v"_a)
      .def("distanceTo",
           [](vec3 &_this, vec3 &pt) { return (_this - pt).norm(); }, "pt"_a);

  v.def("dot", [](vec3 &_this,
                  vec3 &other) { return sofa::defaulttype::dot(_this, other); })
      .def("cross", &vec3::cross<double, 3>, "b"_a);

  v.def("toList", [](const vec3 &v) {
    py::list l;
    for (double i : v)
      l.append(i);
    return l;
  });
  v.def("__str__",
        [](vec3 &v) {
          return std::string("(") + std::to_string(v.x()) + ", " +
                 std::to_string(v.y()) + ", " + std::to_string(v.z()) + ")";
        })
      .def("__repr__", [](vec3 &v) {
        return std::string("vec3(") + std::to_string(v.x()) + ", " +
               std::to_string(v.y()) + ", " + std::to_string(v.z()) + ")";
      });
}
