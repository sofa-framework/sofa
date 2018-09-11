#include "Binding_BaseData.h"

void moduleAddBaseData(py::module &m)
{
  py::class_<BaseData, raw_ptr<BaseData>> p(m, "BaseData");
  p.def("setName", &BaseData::setName);
  p.def("getName", &BaseData::getName);
}
