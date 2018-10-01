#include "Binding_PythonController.h"
#include "Binding_Base.h"
#include <pybind11/detail/init.h>

void PythonController::init() {
    std::cout << " PythonController::init()" << std::endl;
}

void PythonController::reinit() {
    std::cout << " PythonController::reinit()" << std::endl;
}

template <typename T> class py_shared_ptr : public sofa::core::sptr<T>
{
public:
    py_shared_ptr(T *ptr) ;
};

class PythonController_Trampoline : public PythonController
{
private:
    std::shared_ptr<PyObject> o;

public:
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
        return o->ob_type->tp_name;
    }

    void setInstance(py::object s){
        py::print(py::str( s.get_type() ));

        s.inc_ref();

        // TODO(bruno-marques) ici ça crash dans SOFA.
        //--ref_counter;

        o = std::shared_ptr<PyObject>( s.ptr(), [](PyObject* ob)
        {
            // runSofa Sofa/tests/pyfiles/ScriptController.py => CRASH
            // Py_DECREF(ob);
        });
     }

    virtual void init() override ;
    virtual void reinit() override ;
};

void PythonController_Trampoline::init()
{
    //std::cout << "PythonController_trampoline::init()" << std::endl;
    PYBIND11_OVERLOAD(void, PythonController, init, );
}

void PythonController_Trampoline::reinit()
{
    //std::cout << "PythonController_trampoline::reinit()" << std::endl;
    PYBIND11_OVERLOAD(void, PythonController, reinit, );
}

template <typename T>
py_shared_ptr<T>::py_shared_ptr(T *ptr) : sofa::core::sptr<T>(ptr)
{
    auto nptr = dynamic_cast<PythonController_Trampoline*>(ptr);
    if(nptr)
        nptr->setInstance( py::cast(ptr) ) ;
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
