#include "Binding_Mat.h"
#include <functional>
#include <pybind11/operators.h>

#define BINDING_MAT_MAKE_NAME(L, C)                                            \
  std::string(std::string("Mat") + std::to_string(L) + "x" + std::to_string(C))


// Generic bindings for Matrices
template <int L, int C> struct MATRIX {
  static void addMat(py::module &m) {

  }
};

// Prevent bindings with L | C == 0
template <int C> struct MATRIX<0, C> {
  static void addMat(py::module &/*m*/) { }
};
template <int L> struct MATRIX<L, 0> {
  static void addMat(py::module &/*m*/) {}
};

void moduleAddMat(py::module &m) {
  MATRIX<1, 1>::addMat(m);
  MATRIX<2, 2>::addMat(m);
  MATRIX<3, 3>::addMat(m);
  MATRIX<3, 4>::addMat(m);
  MATRIX<4, 4>::addMat(m);
}

