#include "Binding_Base.h"

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

void moduleAddBase(py::module &m)
{
  py::class_<Base, Base::SPtr> p(m, "Base");
  p.def("setName", [](Base& self, const std::string& s){ self.setName(s); });
  p.def("getName", &Base::getName);

  p.def("getData", [](Base& self, const std::string& s) -> py::object
  {
        std::cout << "UP CAST..." ;
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            std::cout << "UP CAST..." ;
            return py::cast(d);
        }
        return py::none();
  });

}
