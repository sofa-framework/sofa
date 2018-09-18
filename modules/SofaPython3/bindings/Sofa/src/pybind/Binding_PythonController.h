#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

class PythonController : public BaseObject
{
public:
    void init() override;
    void reinit() override;
};

template class py::class_<PythonController,
                          BaseObject, sofa::core::sptr<PythonController>>;

void moduleAddPythonController(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
