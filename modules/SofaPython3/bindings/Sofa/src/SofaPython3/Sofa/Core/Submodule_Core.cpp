#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseData.h"
#include "Binding_PythonController.h"
#include "Submodule_Core.h"

#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Context.h>

/// The first parameter must be named the same as the module file to load.
pybind11::module addSubmoduleCore(py::module& p)
{
  py::module core = p.def_submodule("Core");
  moduleAddDataDict(core);
  moduleAddDataDictIterator(core);
  moduleAddBase(core);
  moduleAddBaseData(core);
  //moduleAddDataAsString(core);
  //moduleAddDataAsContainer(core);
  moduleAddBaseObject(core);
  moduleAddPythonController(core);

  py::class_<sofa::core::objectmodel::BaseNode, Base,
          sofa::core::objectmodel::BaseNode::SPtr>(core, "BaseNode");

  py::class_<sofa::core::objectmodel::BaseContext,
          sofa::core::objectmodel::Base,
          sofa::core::objectmodel::BaseContext::SPtr>(core, "BaseContext");

  py::class_<sofa::core::objectmodel::Context,
          sofa::core::objectmodel::BaseContext,
          sofa::core::objectmodel::Context::SPtr>(core, "Context");


  return core;
}
