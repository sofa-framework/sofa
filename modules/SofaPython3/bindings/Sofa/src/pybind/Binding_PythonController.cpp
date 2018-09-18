#include "Binding_PythonController.h"

void PythonController::init() {
  std::cout << "PythonController::init()" << std::endl;
  PYBIND11_OVERLOAD(void, BaseObject, init, );
}

void PythonController::reinit() {
  std::cout << "PythonController::reinit()" << std::endl;
  PYBIND11_OVERLOAD(void, BaseObject, init, );
}

PythonObjectWrapper::PythonObjectWrapper(std::shared_ptr<BaseObject> b) : _b(b) {}
void PythonObjectWrapper::init() { _b->init(); }
void PythonObjectWrapper::reinit() { _b->reinit(); }

void moduleAddPythonController(py::module &m) {
  py::class_<PythonObjectWrapper, PythonObjectWrapper::SPtr> f(m, "Foo");
  f.def(py::init<std::shared_ptr<BaseObject>>());
  f.def("init", &PythonObjectWrapper::init);
  f.def("reinit", &PythonObjectWrapper::reinit);
}
