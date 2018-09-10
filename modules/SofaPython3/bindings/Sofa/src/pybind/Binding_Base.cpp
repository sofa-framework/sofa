#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, boost::intrusive_ptr<Base>, true)

/*class MyBase
{
    public:
        void setName(const std::string& name, int counter){}
        void setName(const std::string& name){}
        const std::string getName(){ return ""; }
};*/

void init_Base(py::module &m)
{
  py::class_<Base, Base::SPtr> p(m, "Base");
  p.def("setName", [](Base& self, const std::string& s){ self.setName(s); });
  p.def("getName", &Base::getName);
}

void registerLoader()
{
    std::cout << "Loader" << std::endl ;
}


PYBIND11_MODULE(example, m) {
    init_Base(m) ;
}


  /*
  p.def_readwrite("distance", &Plane::distance);
  p.def_readwrite("normal", &Plane::normal);
  p.def(py::init<const vec3 &, double>(), "normal"_a = Constants::XAxis,
        "distance"_a = 0);
  p.def(py::init([](py::list l, double dist) {
          return std::unique_ptr<Plane>(
              new Plane(vec3(double(l[0].cast<py::float_>()),
                             double(l[1].cast<py::float_>()),
                             double(l[2].cast<py::float_>())),
                        dist));
        }),
        "normal"_a = Constants::XAxis, "distance"_a = 0);
  p.def(py::init<const vec3 &, const vec3 &>(), "normal"_a, "point"_a);
  p.def("raycast",
        (bool (Plane::*)(const Ray &, double &) const) & Plane::raycast,
        "ray"_a, "p"_a);
  p.def("raycast",
        [](const Plane &plane, Ray r) {
          double p = 0.0;
          plane.raycast(r, p);
          return p;
        },
        "ray"_a);
   */
