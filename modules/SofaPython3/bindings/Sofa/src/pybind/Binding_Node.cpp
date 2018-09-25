
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


    /// The Node::create function will be used as the constructor of the
    /// class two version exists.
    p.def(py::init( [](){ return Node::create("unnamed"); }));
    p.def(py::init( [](const std::string& name){ return Node::create(name); }));

    /// Object's related method. A single addObject is now available
    /// and the createObject is deprecated printing a warning for old scenes.
    p.def("createObject",
          [](Node* self, const std::string& type, const py::kwargs& kwargs)
    {
        return py::cast( simpleapi::createObject(self, type,
                                                 toStringMap(kwargs)) );
    });
    p.def("addObject", [](Node& self, BaseObject* object)
    {
        return self.addObject(object);
    });


    /// Node's related method. A single addNode is now available
    /// and the createChild is deprecated printing a warning for old scenes.
    p.def("createChild", [](Node* self, const std::string& name, const py::kwargs& kwargs)
    {
        return py::cast( simpleapi::createChild(self, name, toStringMap(kwargs)));
    });
    p.def("addChild", [](Node* self, Node* child)
    {
        self->addChild(child);
        return child;
    });
    p.def("getChild", [](Node &n, const std::string &name) -> py::object {
        Node *child = n.getChild(name);
        if (child)
            return py::cast(child);
        return py::none();
    });

    p.def("getRoot", &Node::getRoot);
    p.def("getPath", &Node::getPathName);

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
}
