#include "Binding_Vec.h"
#include <functional>
#include <pybind11/operators.h>

#define BINDING_VEC_MAKE_NAME(N, T)                                            \
  std::string(std::string("Vec") + std::to_string(N) + typeid(T).name())

namespace pyVec {
template <int N, class T> std::string __str__(const Vec<N, T> &v, bool repr) {
  std::string s = (repr) ? (BINDING_VEC_MAKE_NAME(N, T) + "(") : ("(");
  s += std::to_string(v[0]);
  for (size_t i = 1; i < v.size(); ++i)
    s += std::string(", ") + std::to_string(v[i]);
  s += ")";
  return s;
}
} // namespace pyVec

template <int N, class T> void addVec(py::module &m, py::class_<Vec<N, T>> &p) {
  typedef Vec<N, T> VecClass;
  p.def(py::init<>());                 // empty ctor
  p.def(py::init<const VecClass &>()); // copy ctor

  p.def("set", [](VecClass &v, py::list l) { // set
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

  /// Iterator protocol
  static size_t value = 0;
  p.def("__iter__", [](VecClass &v) {
    value = 0;
    return v;
  });
  p.def("__next__", [](VecClass &v) {
    if (value == v.size())
      throw py::stop_iteration();
    else
      return v[value++];
    return v[value];
  });

  p.def(py::self != py::self)
      .def(py::self == py::self)
      .def(py::self * py::self)
      .def(py::self * py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self);

  p.def("__mul__", [](double d, const VecClass &v) { return v * d; });
  p.def("__mul__", [](int d, const VecClass &v) { return v * d; });

  p.def(py::self * float())
      .def(py::self * int())
      .def(py::self *= float())
      .def(py::self *= int());

  p.def(py::self / float())
      .def(py::self / int())
      .def(py::self /= float())
      .def(py::self /= int());

  p.def("__str__", [](VecClass &v) { return pyVec::__str__(v); });
  p.def("__repr__", [](VecClass &v) { return pyVec::__str__(v, true); });

  p.def("fill", &VecClass::fill, "r"_a);
  p.def("clear", &VecClass::clear);
  p.def("norm", &VecClass::norm);
  p.def("norm2", &VecClass::norm2);
  p.def("lNorm", &VecClass::lNorm, "l"_a);
  p.def("normalize", (bool (VecClass::*)(T)) & VecClass::normalize,
        "threshold"_a = std::numeric_limits<T>::epsilon());
  p.def("normalized", &VecClass::normalized);
  p.def("sum", &VecClass::sum);

  m.def("dot",
        (T(*)(const VecClass &a, const VecClass &b)) & sofa::defaulttype::dot);
}

template <int N, class T> struct VEC {
  typedef Vec<N, T> VecClass;
  template <int NN = N, typename std::enable_if<(NN >= 1), int>::type = 0>
  static void add_x(py::class_<VecClass> &p) {
    p.def_property("x", [](VecClass &v) { return v.x(); },
                   [](VecClass &v, double x) { v.x() = x; });
  }
  template <int NN = N, typename std::enable_if<(NN >= 2), int>::type = 0>
  static void add_y(py::class_<VecClass> &p) {
    p.def_property("y", [](VecClass &v) { return v.y(); },
                   [](VecClass &v, double y) { v.y() = y; });
  }
  template <int NN = N, typename std::enable_if<(NN >= 3), int>::type = 0>
  static void add_z(py::class_<VecClass> &p) {
    p.def_property("z", [](VecClass &v) { return v.z(); },
                   [](VecClass &v, double z) { v.z() = z; });
  }
  template <int NN = N, typename std::enable_if<(NN >= 4), int>::type = 0>
  static void add_w(py::class_<VecClass> &p) {
    p.def_property("w", [](VecClass &v) { return v.w(); },
                   [](VecClass &v, double w) { v.w() = w; });
  }

  template <int NN = N, typename std::enable_if<(NN >= 2), int>::type = 0>
  static void add_xy(py::class_<VecClass> &p) {
    p.def_property("xy",
                   [](VecClass &v) {
                     py::tuple t(2);
                     t[0] = v.x();
                     t[1] = v.y();
                     return t;
                   },
                   [](VecClass &v, double x, double y) {
                     v.x() = x;
                     v.y() = y;
                   });
  }

  template <int NN = N, typename std::enable_if<(NN >= 3), int>::type = 0>
  static void add_xyz(py::class_<VecClass> &p) {
    p.def_property("xyz",
                   [](VecClass &v) {
                     py::tuple t(3);
                     t[0] = v.x();
                     t[1] = v.y();
                     t[2] = v.z();
                     return t;
                   },
                   [](VecClass &v, double x, double y, double z) {
                     v.x() = x;
                     v.y() = y;
                     v.z() = z;
                   });
  }

  template <int NN = N, typename std::enable_if<(NN >= 4), int>::type = 0>
  static void add_xyzw(py::class_<VecClass> &p) {
    p.def_property("xyzw",
                   [](VecClass &v) {
                     py::tuple t(4);
                     t[0] = v.x();
                     t[1] = v.y();
                     t[2] = v.z();
                     t[3] = v.w();
                     return t;
                   },
                   [](VecClass &v, double x, double y, double z, double w) {
                     v.x() = x;
                     v.y() = y;
                     v.z() = z;
                     v.w() = w;
                   });
  }
};

// generic bindings
template <int N, class T> struct VECTOR : public VEC<N, T> {
  static void addVec(py::module &m) {
    py::class_<Vec<N, T>> p(m, BINDING_VEC_MAKE_NAME(N, T).c_str());
    ::addVec(m, p);
  }
};

// Prevent calling bindings with N == 0
template <class T> struct VECTOR<0, T> : public VEC<0, T> {
  static void addVec(py::module &m) {}
};

// specializations required for ctors / and variadic template methods
template <class T> struct VECTOR<1, T> : public VEC<1, T> {
  static void addVec(py::module &m) {
    typedef Vec<1, T> VecClass;
    typedef VEC<1, T> PARENT;
    py::class_<VecClass> p(m, BINDING_VEC_MAKE_NAME(1, T).c_str());
    p.def(py::init<T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 1; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T)) & VecClass::set);
    ::addVec(m, p);
    PARENT::add_x(p);
  }
};

template <class T> struct VECTOR<2, T> : public VEC<2, T> {
  static void addVec(py::module &m) {
    typedef Vec<2, T> VecClass;
    typedef VEC<2, T> PARENT;
    py::class_<VecClass> p(m, BINDING_VEC_MAKE_NAME(2, T).c_str());
    p.def(py::init<T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 2; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T)) & VecClass::set);
    ::addVec(m, p);
    PARENT::add_x(p);
    PARENT::add_y(p);
    PARENT::add_xy(p);

    m.def("cross", [](const VecClass &a, const VecClass &b) {
      T val = sofa::defaulttype::cross(a, b);
      return val;
    });
  }
};

template <class T> struct VECTOR<3, T> : public VEC<3, T> {
  static void addVec(py::module &m) {
    typedef VEC<3, T> PARENT;
    typedef Vec<3, T> VecClass;
    py::class_<VecClass> p(m, BINDING_VEC_MAKE_NAME(3, T).c_str());
    p.def(py::init<T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 3; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T)) & VecClass::set);
    m.def("cross", [](const VecClass &a, const VecClass &b) {
      VecClass val = sofa::defaulttype::cross(a, b);
      return val;
    });
    ::addVec(m, p);
    PARENT::add_x(p);
    PARENT::add_y(p);
    PARENT::add_z(p);
    PARENT::add_xy(p);
    PARENT::add_xyz(p);
  }
};

template <class T> struct VECTOR<4, T> : public VEC<4, T> {
  static void addVec(py::module &m) {
    typedef Vec<4, T> VecClass;
    typedef VEC<4, T> PARENT;
    py::class_<VecClass> p(m, BINDING_VEC_MAKE_NAME(4, T).c_str());
    p.def(py::init<T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 4; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
    PARENT::add_x(p);
    PARENT::add_y(p);
    PARENT::add_z(p);
    PARENT::add_w(p);
    PARENT::add_xy(p);
    PARENT::add_xyz(p);
    PARENT::add_xyzw(p);
  }
};

template <class T> struct VECTOR<5, T> : public VEC<5, T> {
  static void addVec(py::module &m) {
    typedef Vec<5, T> VecClass;
    py::class_<Vec<5, T>> p(m, BINDING_VEC_MAKE_NAME(5, T).c_str());
    p.def(py::init<T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 5; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
  }
};

template <class T> struct VECTOR<6, T> : public VEC<6, T> {
  static void addVec(py::module &m) {
    typedef Vec<6, T> VecClass;
    py::class_<Vec<6, T>> p(m, BINDING_VEC_MAKE_NAME(6, T).c_str());
    p.def(py::init<T, T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 6; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
  }
};

template <class T> struct VECTOR<7, T> : public VEC<7, T> {
  static void addVec(py::module &m) {
    typedef Vec<7, T> VecClass;
    py::class_<Vec<7, T>> p(m, BINDING_VEC_MAKE_NAME(7, T).c_str());
    p.def(py::init<T, T, T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 7; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
  }
};

template <class T> struct VECTOR<8, T> : public VEC<8, T> {
  static void addVec(py::module &m) {
    typedef Vec<8, T> VecClass;
    py::class_<Vec<8, T>> p(m, BINDING_VEC_MAKE_NAME(8, T).c_str());
    p.def(py::init<T, T, T, T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 8; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T, T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
  }
};

template <class T> struct VECTOR<9, T> : public VEC<9, T> {
  static void addVec(py::module &m) {
    typedef Vec<9, T> VecClass;
    py::class_<Vec<9, T>> p(m, BINDING_VEC_MAKE_NAME(9, T).c_str());
    p.def(py::init<T, T, T, T, T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 9; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set",
          (void (VecClass::*)(T, T, T, T, T, T, T, T, T)) & VecClass::set);
    ::addVec(m, p);
  }
};

template <class T> struct VECTOR<12, T> : public VEC<12, T> {
  static void addVec(py::module &m) {
    typedef Vec<12, T> VecClass;
    py::class_<Vec<12, T>> p(m, BINDING_VEC_MAKE_NAME(12, T).c_str());
    p.def(py::init<T, T, T, T, T, T, T, T, T, T, T, T>());
    p.def(py::init([](py::list l) {
      VecClass *v = new VecClass();
      for (size_t i = 0; i < 12; ++i) {
        if (std::string(typeid(T).name()) == "i")
          (*v)[i] = int(l[i].cast<py::int_>());
        else
          (*v)[i] = double(l[i].cast<py::float_>());
      }
      return std::unique_ptr<VecClass>(v);
    }));
    p.def("set", (void (VecClass::*)(T, T, T, T, T, T, T, T, T, T, T, T)) &
                     VecClass::set);
    ::addVec(m, p);
  }
};

void moduleAddVec(py::module &m) {

  VECTOR<1, int>::addVec(m);
  VECTOR<2, int>::addVec(m);
  VECTOR<3, int>::addVec(m);
  VECTOR<4, int>::addVec(m);
  VECTOR<5, int>::addVec(m);
  VECTOR<6, int>::addVec(m);
  VECTOR<7, int>::addVec(m);
  VECTOR<8, int>::addVec(m);
  VECTOR<9, int>::addVec(m);
  VECTOR<12, int>::addVec(m);
  VECTOR<1, double>::addVec(m);
  VECTOR<2, double>::addVec(m);
  VECTOR<3, double>::addVec(m);
  VECTOR<4, double>::addVec(m);
  VECTOR<5, double>::addVec(m);
  VECTOR<6, double>::addVec(m);
  VECTOR<7, double>::addVec(m);
  VECTOR<8, double>::addVec(m);
  VECTOR<9, double>::addVec(m);
  VECTOR<12, double>::addVec(m);
}
