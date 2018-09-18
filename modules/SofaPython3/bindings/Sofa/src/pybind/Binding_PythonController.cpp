#include "Binding_PythonController.h"

void PythonController::init()
{
    std::cout << "PythonController::init()" << std::endl;
}

void PythonController::reinit()
{
    std::cout << "PythonController::reinit()" << std::endl;
}

void moduleAddPythonController(py::module& m)
{
    py::class_<PythonController, BaseObject,
               sofa::core::sptr<PythonController>> p(m, "PythonController");

    p.def(py::init());
}
