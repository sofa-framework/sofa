#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

class PythonController : public BaseObject {
public:
  void init() override;
  void reinit() override;
};

struct PythonObjectWrapper : public BaseObject {
  SOFA_CLASS(PythonObjectWrapper, BaseObject);
  std::shared_ptr<BaseObject> _b;

  PythonObjectWrapper(std::shared_ptr<BaseObject> b);
  void init();
  void reinit();
};

void moduleAddPythonController(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
