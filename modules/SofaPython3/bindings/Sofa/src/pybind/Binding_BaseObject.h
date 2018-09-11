#ifndef PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H
#define PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H

#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

template class py::class_<BaseObject, Base, sofa::core::sptr<BaseObject>>;

void moduleAddBaseObject(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_BASEOBJECT_H
