#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseData.h"
#include "Binding_Node.h"
#include "Binding_PythonController.h"
#include "Binding_Simulation.h"

/// The first parameter must be named the same as the module file to load.
PYBIND11_MODULE(Sofa, m)
{
  moduleAddBase(m);
  moduleAddBaseData(m);
  moduleAddDataAsString(m);
  moduleAddDataAsContainer(m);
  moduleAddBaseObject(m);
  moduleAddNode(m);
  moduleAddPythonController(m);
  moduleAddSimulation(m);
}
