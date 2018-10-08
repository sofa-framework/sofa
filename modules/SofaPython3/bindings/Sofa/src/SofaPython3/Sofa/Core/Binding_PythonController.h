#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_BaseObject.h"

namespace sofapython3
{
class PythonController : public BaseObject
{
public:
    SOFA_CLASS(PythonController, BaseObject);
    void init() override ;
    void reinit() override;

    PythonController()
    {
        std::cout << "PythonController() at "<<(void*)this << std::endl;
    }

    ~PythonController()
    {
        std::cout << "~PythonController()" << std::endl;

    }
};


template <typename T> class py_shared_ptr : public sofa::core::sptr<T>
{
public:
    py_shared_ptr(T *ptr) ;
};


void moduleAddPythonController(py::module &m);

} /// namespace sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
