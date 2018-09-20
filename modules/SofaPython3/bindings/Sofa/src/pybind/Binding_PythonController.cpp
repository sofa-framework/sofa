#include "Binding_PythonController.h"
#include "Binding_Base.h"
#include <pybind11/detail/init.h>
namespace pybind11
{
namespace detail {
namespace initimpl {
// Implementing class for py::init_alias<...>()
template <typename... Args> struct salias_constructor {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<Alias<Class>, Args...>::value, int> = 0>

    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            std::cout << "INIT" << std::endl ;

            v_h.value_ptr() = construct_or_initialize<Alias<Class>>(std::forward<Args>(args)...);
            std::cout << "INIT" << std::endl ;
        }, is_new_style_constructor(), extra...);
    }
};
}
}

template <typename... Args> detail::initimpl::salias_constructor<Args...> init_salias() { return {}; }

//template <typename... Args, typename... Extra>
//class_ &def(const detail::initimpl::salias_constructor<Args...> &init, const Extra&... extra) {
//    init.execute(*this, extra...);
//    return *this;
//}
}

template <typename T> class py_shared_ptr
{
public:
    py::object pyobj ;
    sofa::core::sptr<T> sofaobj;

    py_shared_ptr(T *ptr)
    {
        sofaobj = sofa::core::sptr<T>(ptr);
        pyobj = py::cast(ptr);

        std::cout << "REF " << pyobj.ref_count() << std::endl ;
        std::cout << "construct (" << (void*)pyobj.ptr() << ", " << (void*)ptr << ")" <<  std::endl ;
    }

    ~py_shared_ptr()
    {
        //py::object pyobj = py::cast(sofa::core::sptr<T>::get());
       std::cout << "destruct " << std::endl ;
    }

};


void PythonController::init() {
    std::cout << " PythonController::init()" << std::endl;
}

void PythonController::reinit() {
    std::cout << " PythonController::reinit()" << std::endl;
}

class PythonController_Trampoline : public PythonController
{
public:
    PyObject* o {nullptr};
    PythonController_Trampoline()
    {
        std::cout << "PythonController_Trampoline() at "<<(void*)this<<std::endl;
    }

    ~PythonController_Trampoline()
    {
        std::cout << "~PythonController_Trampoline()"<<std::endl;
    }

    virtual std::string getClassName() const override
    {
        return "PythonController";
    }

    virtual void __init__() override
    {
        std::cout << "PythonController_trampoline::__" << std::endl;
        PYBIND11_OVERLOAD(void, PythonController, __init__, );
    }

    virtual void init() override ;
    virtual void reinit() override ;
};

void PythonController_Trampoline::init()
{
    std::cout << "PythonController_trampoline::init()" << std::endl;
    PYBIND11_OVERLOAD(void, PythonController, init, );
}

void PythonController_Trampoline::reinit()
{
    std::cout << "PythonController_trampoline::reinit()" << std::endl;
    PYBIND11_OVERLOAD(void, PythonController, reinit, );
}

PYBIND11_DECLARE_HOLDER_TYPE(PythonController,
                             py_shared_ptr<PythonController>, true)

void moduleAddPythonController(py::module &m) {
    py::class_<PythonController, BaseObject,
            PythonController_Trampoline,
            py_shared_ptr<PythonController>> f(m, "PythonController");

    f.def(py::init());
    f.def("init", &PythonController::init);
    f.def("reinit", &PythonController::reinit);
}
