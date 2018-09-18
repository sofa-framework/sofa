
#include <pybind11/pybind11.h>
//#include <src/pybind/Binding_BoundingBox.h>
//#include <src/pybind/Binding_Color.h>
//#include <src/pybind/Binding_Frame.h>
#include <src/pybind/Binding_Mat.h>
#include <src/pybind/Binding_Quat.h>
#include <src/pybind/Binding_Vec.h>

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(SofaTypes, m) {
//  moduleAddBoundingBox(m);
//  moduleAddColor(m);
//  moduleAddFrame(m);
  moduleAddMat(m);
  moduleAddQuat(m);
  moduleAddVec(m);
}
