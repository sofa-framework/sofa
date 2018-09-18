#include "Binding_PythonController.h"

void PythonController::init()
{
    std::cout << "PythonController::init()" << std::endl;
}

void PythonController::reinit()
{
    std::cout << "PythonController::reinit()" << std::endl;
}

class PythonController_PyExt : public PythonController
{
public:
    void init() override
    {
        PYBIND11_OVERLOAD(
            void,                       /* Return type */
            PythonController,           /* Parent class */
            init                        /* Name of function in C++ (must match Python name) */
        );
    }

    void reinit() override
    {
        PYBIND11_OVERLOAD(
            void,                       /* Return type */
            PythonController,           /* Parent class */
            reinit                        /* Name of function in C++ (must match Python name) */
        );
    }
};

void moduleAddPythonController(py::module& m)
{
    py::class_<PythonController, BaseObject,
               PythonController_PyExt,
               sofa::core::sptr<PythonController>> p(m, "PythonController", py::dynamic_attr());


    p.def(py::init());

    /*p.def("__setattr__", [](Base& self, const std::string& s, py::object& value) -> py::object
    {
        std::cout << "SET ATTR LOCAL: " << s << std::endl ;
        return py::none();
    });*/

}
