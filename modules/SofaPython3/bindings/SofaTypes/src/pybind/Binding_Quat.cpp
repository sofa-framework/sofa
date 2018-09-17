#include "Binding_Quat.h"
typedef sofa::helper::Quater<double> Quat;
#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Vec4d;
#include <sofa/defaulttype/Mat.h>
typedef sofa::defaulttype::Mat4x4d Matrix4;
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
  p.def_static("set", [](double a0, double a1, double a2) {
    return Quat::set(a0, a1, a2);
  });
  p.def_static("set", [](Vec3d V) { return Quat::set(V); });
  p.def("identity", &Quat::identity);

  p.def("normalize", &Quat::normalize);
  p.def("clear", &Quat::clear);
  p.def("fromFrame", &Quat::fromFrame, "x"_a, "y"_a, "z"_a);
  p.def("fromMatrix", &Quat::fromMatrix, "m"_a);
//  p.def("toMatrix", &Quat::toMatrix, "m"_a);
//  p.def("rotate", &Quat::rotate, "v"_a);
//  p.def("inverseRotate", &Quat::inverseRotate, "v"_a);
  p.def("inverse", &Quat::inverse);
  p.def("toRotationVector", &Quat::quatToRotationVector);
  p.def("toEulerVector", &Quat::toEulerVector);
  //  p.def("slerp", &Quat::slerp, "a"_a, "b"_a, "t"_a, "allowdFlip"_a = true);
  p.def("buildRotationMatrix", [](Quat &self, Matrix4 &m) {
    double tmp[4][4] = {0};
    self.buildRotationMatrix(tmp);
    m = Matrix4(Vec4d(tmp[0]), Vec4d(tmp[1]), Vec4d(tmp[2]), Vec4d(tmp[3]));
  });
  p.def("axisToQuat", &Quat::axisToQuat, "a"_a, "phi"_a);
  p.def("quatToAxis", &Quat::quatToAxis, "a"_a, "phi"_a);
//  p.def_static("createFromFrame", &Quat::createFromFrame);
//  p.def_static("createFromRotationVector",
//               (Quat(Quat::*)(Vec3d)) & Quat::createFromRotationVector);
//  p.def_static("createFromRotationVector",
//               (Quat(Quat::*)(double, double, double)) &
//                   Quat::createFromRotationVector);
//  p.def_static("createFromEuler",
//               (Quat(Quat::*)(Vec3d)) & Quat::createFromEuler);
//  p.def_static("createFromEuler",
//               (Quat(Quat::*)(double, double, double)) & Quat::createFromEuler);
  p.def("size", &Quat::size);

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
}
