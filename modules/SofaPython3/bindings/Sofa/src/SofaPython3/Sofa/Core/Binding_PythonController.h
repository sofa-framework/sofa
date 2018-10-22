#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_BaseObject.h"

#include <sofa/core/behavior/BaseController.h>

template class pybind11::class_<sofa::core::behavior::BaseController,
                          sofa::core::objectmodel::BaseObject,
                          sofa::core::sptr<sofa::core::behavior::BaseController>>;


namespace sofapython3
{
using sofa::core::behavior::BaseController;

class PythonController : public BaseController
{
public:
    SOFA_CLASS(PythonController, BaseController);
    void init() override ;
    void reinit() override;

    PythonController()
    {
    }

    ~PythonController()
    {
    }
};

void moduleAddPythonController(py::module &m);

} /// namespace sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
