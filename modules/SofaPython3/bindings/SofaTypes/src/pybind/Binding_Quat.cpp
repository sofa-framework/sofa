#include "Binding_Quat.h"
typedef sofa::helper::Quater<double> Quat;
#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Vec4d;
#include <sofa/defaulttype/Mat.h>
typedef sofa::defaulttype::Mat4x4d Matrix4;
typedef sofa::defaulttype::Mat3x3d Matrix3;
#include <pybind11/operators.h>

void moduleAddQuat(py::module &m) {
  py::class_<Quat> p(m, "Quat");
  p.def(py::init());
  p.def(py::init<double, double, double, double>(), "x"_a, "y"_a, "z"_a, "w"_a);
  p.def(py::init<Quat>());
  p.def(py::init<Vec3d, double>(), "axis"_a, "angle"_a);
  p.def(py::init([](py::list l) {
    Quat *q = new Quat();
    for (size_t i = 0; i < 4; ++i) {
      (*q)[i] = double(l[i].cast<py::float_>());
    }
    return std::unique_ptr<Quat>(q);
  }));
  p.def(py::init<Vec3d, Vec3d>(), "vFrom"_a, "vTo"_a);

  p.def("set", (void (Quat::*)(double, double, double, double)) & Quat::set,
        "x"_a, "y"_a, "z"_a, "w"_a);
  p.def("identity", &Quat::identity);

  p.def("normalize", &Quat::normalize);
  p.def("clear", &Quat::clear);
  p.def("fromFrame", &Quat::fromFrame, "x"_a, "y"_a, "z"_a);
  p.def("fromMatrix", &Quat::fromMatrix, "m"_a);
  p.def("toMatrix", [](Quat &self, Matrix3 &m) { return self.toMatrix(m); },
        "m"_a);
  p.def("rotate", [](Quat &self, const Vec3d &v) { return self.rotate(v); },
        "v"_a);
  p.def("inverseRotate",
        [](Quat &self, const Vec3d &v) { return self.inverseRotate(v); },
        "v"_a);
  p.def("inverse", &Quat::inverse);
  p.def("toRotationVector", &Quat::quatToRotationVector);
  p.def("toEulerVector", &Quat::toEulerVector);
  p.def("buildRotationMatrix", [](Quat &self, Matrix4 &m) {
    double tmp[4][4] = {0};
    self.buildRotationMatrix(tmp);
    m = Matrix4(Vec4d(tmp[0]), Vec4d(tmp[1]), Vec4d(tmp[2]), Vec4d(tmp[3]));
  });
  p.def("axisToQuat", &Quat::axisToQuat, "a"_a, "phi"_a);
  p.def("quatToAxis", &Quat::quatToAxis, "a"_a, "phi"_a);
  p.def_static("createFromFrame", &Quat::createQuaterFromFrame);
  p.def_static("createFromRotationVector",
               [](const Vec3d &a) { Quat::createFromRotationVector(a); });
  p.def_static("createFromRotationVector", [](double a0, double a1, double a2) {
    Quat::createFromRotationVector(a0, a1, a2);
  });
  p.def_static("createFromEuler",
               [](Vec3d v) { Quat::createQuaterFromEuler(v); });
  p.def_static("createFromEuler", [](double alpha, double beta, double gamma) {
    Quat::fromEuler(alpha, beta, gamma);
  });
  p.def("size", &Quat::size);

  p.def("slerp",
        (void (Quat::*)(const Quat &, const Quat &, double, bool)) &
            Quat::slerp,
        "a"_a, "b"_a, "t"_a, "allowdFlip"_a = true);
  p.def("slerp", (Quat(Quat::*)(Quat &, double)) & Quat::slerp, "q1"_a, "t"_a);
  p.def("slerp2", &Quat::slerp2, "q1"_a, "t"_a);

  p.def(py::self + py::self);
  p.def(py::self * py::self);
  p.def(py::self * double());
  p.def(py::self / double());
  p.def(py::self *= double());
  p.def(py::self /= double());

  p.def(py::self += py::self);
  p.def(py::self *= py::self);
  p.def(py::self == py::self);
  p.def(py::self != py::self);

  p.def("__getitem__",
        [](const Quat &self, size_t i) {
          if (i >= self.size())
            throw py::index_error();
          return self[i];
        })
      .def("__setitem__", [](Quat &self, size_t i, double d) {
        if (i >= self.size())
          throw py::index_error();
        double &val = self[i];
        val = d;
        return val;
      });

  /// Iterator protocol
  static size_t value = 0;
  p.def("__iter__", [](Quat &self) {
    value = 0;
    return self;
  });
  p.def("__next__", [](Quat &self) {
    if (value == self.size())
      throw py::stop_iteration();
    else
      return self[value++];
    return self[value];
  });

  p.def("__str__", [](Quat &self) {
    std::string s("(");
    s += std::to_string(self[0])
            + ", " + std::to_string(self[1])
            + ", " + std::to_string(self[2])
            + ", " + std::to_string(self[3])
            + ")";
    return s;
  });
  p.def("__repr__", [](Quat &self) {
      std::string s("Quat(");
      s += std::to_string(self[0])
              + ", " + std::to_string(self[1])
              + ", " + std::to_string(self[2])
              + ", " + std::to_string(self[3])
              + ")";
      return s;
  });

}
