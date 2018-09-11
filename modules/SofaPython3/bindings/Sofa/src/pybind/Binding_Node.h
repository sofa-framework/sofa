#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

/// More info about smart pointer in
/// /pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
PYBIND11_DECLARE_HOLDER_TYPE(Base, boost::intrusive_ptr<Base>, true)


#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

template class py::class_<Base, Base::SPtr>;
template class py::class_<Node, Base, Node::SPtr>;

