#ifndef PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H
#define PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H

#include <pybind11/pybind11.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include "Binding_Base.h"
#include "Binding_BaseObject.h"

template class pybind11::class_<sofa::core::objectmodel::BaseObject,
                          sofa::core::objectmodel::Base,
                          sofa::core::sptr<sofa::core::objectmodel::BaseObject>>;

namespace sofapython3
{
using sofa::core::objectmodel::BaseObject;

void moduleAddBaseObject(py::module &m);
} /// namespace sofapython

#endif /// PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H
