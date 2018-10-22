
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/objectmodel/Context.h>

#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseData.h"
#include "Binding_DataContainer.h"
#include "Binding_ForceField.h"
#include "Binding_PythonController.h"
#include "Submodule_Core.h"



namespace sofapython3
{

/// The first parameter must be named the same as the module file to load.
pybind11::module addSubmoduleCore(py::module& p)
{   
    py::module core = p.def_submodule("Core");

    moduleAddDataDict(core);
    moduleAddDataDictIterator(core);
    moduleAddBase(core);
    moduleAddBaseData(core);
    moduleAddWriteAccessor(core);
    moduleAddDataContainer(core);
    moduleAddBaseObject(core);
    moduleAddPythonController(core);
    moduleAddForceField(core);

    py::class_<sofa::core::objectmodel::BaseNode,
            sofa::core::objectmodel::Base,
            sofa::core::objectmodel::BaseNode::SPtr>(core, "BaseNode");

    py::class_<sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Base,
            sofa::core::objectmodel::BaseContext::SPtr>(core, "BaseContext");

    py::class_<sofa::core::objectmodel::Context,
            sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Context::SPtr>(core, "Context");

    return core;
}

} ///namespace sofapython3
