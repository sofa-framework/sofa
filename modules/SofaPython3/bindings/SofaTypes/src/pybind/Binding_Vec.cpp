#include "Binding_Vec.h"
#include <pybind11/operators.h>

#define BINDING_VEC_MAKE_NAME(N, type)                                         \
  std::string(std::string("Vec") + std::to_string(N) + typeid(type).name())
//#define BINDING_ARRAY_MAKE_NAME(N, type)                                       \
//  std::string(std::string("array") + std::to_string(N) + typeid(type).name())

template <int N, class T> struct VECTOR {
  typedef Vec<N, T> VecClass;
  //  typedef sofa::helper::fixed_array<T, N> Array;

  static const int total_size = N;

  static void addVec(py::module &m, T type = 0) {
    //    py::class_<Array>(m, BINDING_ARRAY_MAKE_NAME(N, type).c_str());
    py::class_<VecClass /*, Array*/> p(m,
                                       BINDING_VEC_MAKE_NAME(N, type).c_str());
    p.def(py::init<>());

    // @damienmarchal: any idea?
//    if (total_size == 1)
//        p.def("set", [](VecClass &v, T _1) { v.set(_1); });
//    if (total_size == 2)
//        p.def("set", [](VecClass &v, T _1, T _2) { v.set(_1, _2); });
//    if (total_size == 3)
//        p.def("set", (void (VecClass::*)(T, T, T)) & VecClass::set);
//    if (total_size == 4)
//        p.def("set", (void (VecClass::*)(T, T, T, T)) & VecClass::set);
//    if (total_size == 5)
//        p.def("set", (void (VecClass::*)(T, T, T, T, T)) & VecClass::set);
//    if (total_size == 6)
//        p.def("set", (void (VecClass::*)(T, T, T, T, T, T)) &
//              VecClass::set);
//    if (total_size == 7)
//        p.def("set", (void (VecClass::*)(T, T, T, T, T, T, T)) &
//              VecClass::set);
//    if (total_size == 8)
//        p.def("set",
//              (void (VecClass::*)(T, T, T, T, T, T, T, T)) & VecClass::set);
//    if (total_size == 9)
//        p.def("set",
//              (void (VecClass::*)(T, T, T, T, T, T, T, T, T)) &
//              VecClass::set);
//    if (total_size == 12)
//        p.def("set", (void (VecClass::*)(T, T, T, T, T, T, T, T, T, T, T,
//                                         T)) &
//              VecClass::set);

    p.def("set", [](VecClass &v, py::list l) {
      for (size_t i = 0; i < N && i < l.size(); ++i) {
        T &val = v[i];
        if (std::string(typeid(T).name()) == "i")
          val = int(l[i].cast<py::int_>());
        else
          val = double(l[i].cast<py::float_>());
      }
    });

    p.def("__getitem__",
          [](const VecClass &v, size_t i) {
            if (i >= v.size())
              throw py::index_error();
            return v[i];
          })
        .def("__setitem__", [](VecClass &v, size_t i, T d) {
          if (i >= v.size())
            throw py::index_error();
          T &val = v[i];
          val = d;
          return val;
        });
    p.def(py::self != py::self)
        .def(py::self * py::self)
        .def(py::self * float())
        .def(py::self *= float())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self);
    p.def("__str__", [](VecClass &v) {
      std::string s("(");
      s += std::to_string(v[0]);
      for (size_t i = 1; i < v.size(); ++i)
        s += std::string(", ") + std::to_string(v[i]);
      s += ")";
      return s;
    });
    p.def("__repr__", [type](VecClass &v) {
      std::string s = BINDING_VEC_MAKE_NAME(N, type) + "(";
      s += std::to_string(v[0]);
      for (size_t i = 1; i < v.size(); ++i)
        s += std::string(", ") + std::to_string(v[i]);
      s += ")";
      return s;
    });
  }
};

template <class T> struct VECTOR<0, T> {
  static void addVec(py::module &m, T type = 0) { SOFA_UNUSED(type); }
};

void moduleAddVec(py::module &m) {
  VECTOR<1, int>::addVec(m);
  VECTOR<2, int>::addVec(m);
  VECTOR<3, int>::addVec(m);
  VECTOR<4, int>::addVec(m);
  VECTOR<5, int>::addVec(m);
  VECTOR<6, int>::addVec(m);
  VECTOR<1, double>::addVec(m);
  VECTOR<2, double>::addVec(m);
  VECTOR<3, double>::addVec(m);
  VECTOR<4, double>::addVec(m);
  VECTOR<5, double>::addVec(m);
  VECTOR<6, double>::addVec(m);
}
