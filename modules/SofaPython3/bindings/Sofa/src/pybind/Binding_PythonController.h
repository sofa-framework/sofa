#ifndef PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
#define PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H

#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

class PythonController : public BaseObject {
public:
  void init() override;
  void reinit() override;

  PyObject* o;

  PythonController(){
      std::cout << "PythonController()... " << std::endl;
  }

  ~PythonController(){
      Py_DECREF(o);
      std::cout << "Delete PythonController... " << std::endl;
  }

  void setPythonInstance(PyObject* o){
      this->o = o;
  }

  friend inline void intrusive_ptr_add_ref(PythonController* p)
  {
      std::cout << "ADD REF" << std::endl;
  }

  friend inline void intrusive_ptr_release(PythonController* p)
  {
      std::cout << "DEL REF" << std::endl;
  }
};

/*
struct PythonObjectWrapper : public BaseObject {
  SOFA_CLASS(PythonObjectWrapper, BaseObject);
  std::shared_ptr<BaseObject> _b;

  PythonObjectWrapper(std::shared_ptr<BaseObject> b);
  void init();
  void reinit();
};*/

void moduleAddPythonController(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_PYTHONCONTROLLER_H
