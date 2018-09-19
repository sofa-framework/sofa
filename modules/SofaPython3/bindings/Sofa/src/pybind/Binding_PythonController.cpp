#include "Binding_PythonController.h"
#include "Binding_Base.h"

void PythonController::init() {
    std::cout << "PythonController::init()" << std::endl;
    PYBIND11_OVERLOAD(void, BaseObject, init, );
}

void PythonController::reinit() {
    std::cout << "PythonController::reinit()" << std::endl;
    PYBIND11_OVERLOAD(void, BaseObject, init, );
}

/*
PythonObjectWrapper::PythonObjectWrapper(std::shared_ptr<BaseObject> b) : _b(b) {}
void PythonObjectWrapper::init() {
    std::cout << "PythonObjectWrapper::init()" << std::endl;
    _b->init();
}

void PythonObjectWrapper::reinit() {
    std::cout << "PythonObjectWrapper::reinit()" << std::endl;
    _b->reinit();
}*/

void moduleAddPythonController(py::module &m) {
    /*py::class_<PythonObjectWrapper, PythonObjectWrapper::SPtr> f(m, "Foo");
  f.def(py::init<std::shared_ptr<BaseObject>>());
  f.def("init", &PythonObjectWrapper::init);
  f.def("reinit", &PythonObjectWrapper::reinit);*/

    py::class_<PythonController, BaseObject, sofa::core::sptr<PythonController>> f(m, "PythonController");
    f.def(py::init());
    f.def("init", &PythonController::init);
    f.def("reinit", &PythonController::reinit);
}
