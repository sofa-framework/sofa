
/// Neede to have automatic conversion from pybind types to stl container.
#include <pybind11/stl.h>

#include "Binding_Base.h"
#include "Binding_Node.h"

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Simulation.h>
using sofa::core::objectmodel::BaseData;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi = sofa::simpleapi;

#include "Binding_PythonController.h"

std::string toSofaParsableString(const py::handle& p)
{
    if(py::isinstance<py::list>(p) || py::isinstance<py::tuple>(p))
    {
        std::stringstream tmp;
        for(auto pa : p){
            tmp << toSofaParsableString(pa) << " ";
        }
        return tmp.str();
    }
    //TODO(dmarchal) This conversion to string is so bad.
    if(py::isinstance<py::str>(p))
        return py::str(p);
    return py::repr(p);
}

/// RVO optimized function. Don't care about copy on the return code.
std::map<std::string, std::string> toStringMap(const py::dict& dict)
{
    std::map<std::string, std::string> tmp;
    for(auto kv : dict)
    {
        tmp[py::str(kv.first)] = toSofaParsableString(kv.second);
    }
    return tmp;
}

void moduleAddNode(py::module &m) {
    py::class_<sofa::core::objectmodel::BaseNode, Base,
            sofa::core::objectmodel::BaseNode::SPtr>(m, "BaseNode");

    py::class_<sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Base,
            sofa::core::objectmodel::BaseContext::SPtr>(m, "BaseContext");

    py::class_<sofa::core::objectmodel::Context,
            sofa::core::objectmodel::BaseContext,
            sofa::core::objectmodel::Context::SPtr>(m, "Context");

    py::class_<Node, sofa::core::objectmodel::BaseNode,
            sofa::core::objectmodel::Context, Node::SPtr>
            p(m, "Node");

    p.def("createObject",
          [](Node& self, const std::string& type, const py::kwargs& kwargs) -> py::object
    {
        return py::cast( simpleapi::createObject(&self, type,
                                                 toStringMap(kwargs)) );
    });

    p.def("createChild", &Node::createChild);
    p.def("getRoot", &Node::getRoot);

    p.def("addObject", [](Node& self, py::object object)
    {
        return self.addObject(py::cast<BaseObject*>(object));
    });

    p.def("__getattr__", [](Node& self, const std::string& name) -> py::object
    {
        BaseObject *object = self.getObject(name);
        if (object)
            return py::cast(object);

        Node *child = self.getChild(name);
        if (child)
            return py::cast(child);

        return BindingBase::GetAttr(self, name);
    });

    p.def("getChild", [](Node &n, const std::string &name) -> py::object {
        sofa::simulation::Node *child = n.getChild(name);
        if (child)
            return py::cast(child);
        else
            return py::none();
    });
}
