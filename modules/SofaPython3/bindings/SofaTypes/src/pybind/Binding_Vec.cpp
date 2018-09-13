#include "Binding_Vec.h"

template <int N, class T> struct VECTOR {
  static void addVec(py::module &m, T *type = nullptr) { SOFA_UNUSED(type); }
};

template <int N> struct VECTOR<N, int> {
  static void addVec(py::module &m, int * /*type */ = nullptr) {
    py::class_<Vec<N, int>> v(
        m,
        std::string(std::string("Vec") + std::to_string(N) + typeid(int).name())
            .c_str()); // ....
    // static_asserts handle limit for the number of args..
    // there MUST be a cleaner way of doing this...
    switch (N) {
    case 1:
      v.def(py::init<int>());
      break;
    case 2:
      v.def(py::init<int, int>());
      break;
    case 3:
      v.def(py::init<int, int, int>());
      break;
    case 4:
      v.def(py::init<int, int, int, int>());
      break;
    case 5:
      v.def(py::init<int, int, int, int, int>());
      break;
    case 6:
      v.def(py::init<int, int, int, int, int, int>());
      break;
    };
    // does not compile.. why?
    v.def(py::init<>([](py::list l) {
      std::unique_ptr<Vec<N, int>> vec(new Vec<N, int>());
      for (size_t i = 0; i < N; ++i) {
        int &val = (*(vec.get()))[i];
        val = int(l[i].cast<py::int_>());
      }
    }));
  }
};

template <int N> struct VECTOR<N, double> {
  static void addVec(py::module &m, double * /*type */ = nullptr) {
    py::class_<Vec<N, double>> v(m, std::string(std::string("Vec") +
                                                std::to_string(N) +
                                                typeid(double).name())
                                        .c_str());
    switch (N) {
    case 1:
      v.def(py::init<double>());
      break;
    case 2:
      v.def(py::init<double, double>());
      break;
    case 3:
      v.def(py::init<double, double, double>());
      break;
    case 4:
      v.def(py::init<double, double, double, double>());
      break;
    case 5:
      v.def(py::init<double, double, double, double, double>());
      break;
    case 6:
      v.def(py::init<double, double, double, double, double, double>());
      break;
    };
    v.def(py::init<>([](py::list l) {
      std::unique_ptr<Vec<N, double>> vec(new Vec<N, double>());
      for (size_t i = 0; i < N; ++i) {
        double &val = (*(vec.get()))[i];
        val = double(l[i].cast<py::float_>());
      }
    }));
  }
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
