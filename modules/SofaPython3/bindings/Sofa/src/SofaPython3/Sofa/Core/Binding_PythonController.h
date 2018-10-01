#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_BaseObject.h"

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

void moduleAddPythonController(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
