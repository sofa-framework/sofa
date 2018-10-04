
/// Neede to have automatic conversion from pybind types to stl container.
#include <pybind11/stl.h>

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Simulation.h>
using sofa::core::objectmodel::BaseData;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi = sofa::simpleapi;

#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaSimulationGraph/DAGNode.h>

#include <SofaPython3/Sofa/Core/Binding_Base.h>
#include "Binding_Node.h"

using sofa::core::objectmodel::BaseObjectDescription;

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
void fillBaseObjectdescription(BaseObjectDescription& desc, const py::dict& dict)
{
    for(auto kv : dict)
    {
        desc.setAttribute(py::str(kv.first), toSofaParsableString(kv.second));
    }
}

bool checkParamUsage(Base* object, BaseObjectDescription& desc)
{
    bool hasFailure = false;
    std::stringstream tmp;
    tmp <<"Unknown Attribute(s): " << msgendl;
    for( auto it : desc.getAttributeMap() )
    {
        if (!it.second.isAccessed())
        {
            hasFailure = true;
            tmp << " - \""<<it.first <<"\" with value: \"" <<(std::string)it.second << msgendl;
        }
    }
    if(!desc.getErrors().empty())
    {
        hasFailure = true;
        tmp << desc.getErrors()[0];
    }
    if(hasFailure)
        throw py::type_error(tmp.str());
    return hasFailure;
}

void moduleAddNodeIterator(py::module &m)
{
    py::class_<NodeIterator> d(m, "NodeIterator");

    d.def("__getitem__", [](NodeIterator& d, size_t index) -> py::object
    {
        if(index>=d.size(d.owner.get()))
            throw py::index_error("Too large index '"+std::to_string(index)+"'");
        return py::cast(d.get(d.owner.get(), index));
    });

    d.def("__iter__", [](NodeIterator& d)
    {
        return d;
    });
    d.def("__next__", [](NodeIterator& d) -> py::object
    {
        if(d.index>=d.size(d.owner.get()))
            throw py::stop_iteration();
        return py::cast(d.get(d.owner.get(), d.index++));
    });
    d.def("__len__", [](NodeIterator& d) -> py::object
    {
        return py::cast(d.size(d.owner.get()));
    });
}


void moduleAddNode(py::module &m) {
    moduleAddNodeIterator(m);

    py::class_<Node, sofa::core::objectmodel::BaseNode,
            sofa::core::objectmodel::Context, Node::SPtr>
            p(m, "Node");


    /// The Node::create function will be used as the constructor of the
    /// class two version exists.
    p.def(py::init([](){ return sofa::simulation::graph::DAGNode::SPtr(); }));
    p.def(py::init([](const std::string& name){
        return sofa::core::objectmodel::New<sofa::simulation::graph::DAGNode>(name);
    }));

    /// Object's related method. A single addObject is now available
    /// and the createObject is deprecated printing a warning for old scenes.
    p.def("createObject",
          [](Node* self, const std::string& type, const py::kwargs& kwargs)
    {
        BaseObjectDescription desc {type.c_str(), type.c_str()};
        fillBaseObjectdescription(desc, kwargs);
        auto object = simpleapi::createObject(self, type, desc);
        if(object)
            checkParamUsage(object.get(), desc);
        return py::cast(object);
    });
    p.def("addObject", [](Node& self, BaseObject* object)
    {
        return self.addObject(object);
    });


    /// Node's related method. A single addNode is now available
    /// and the createChild is deprecated printing a warning for old scenes.
    p.def("createChild", [](Node* self, const std::string& name, const py::kwargs& kwargs)
    {
        BaseObjectDescription desc (name.c_str());
        fillBaseObjectdescription(desc,kwargs);
        auto node=simpleapi::createChild(self, name, desc);
        checkParamUsage(node.get(), desc);
        return py::cast(node);
    });

    p.def("addChild", [](Node* self, Node* child)
    {
        self->addChild(child);
        return child;
    });

    p.def("getChild", [](Node &n, const std::string &name) -> py::object
    {
        Node *child = n.getChild(name);
        if (child)
            return py::cast(child);
        return py::none();
    });

    p.def("removeChild", [](Node* n, Node* n2)
    {
        n->removeChild(n2);
        return py::cast(n2);
    });

    p.def("removeChild", [](Node& n, const std::string name)
    {
        Node* node = n.getChild(name);
        if(node==nullptr)
            throw py::index_error("Invalid name '"+name+"'");

        n.removeChild(node);
        return py::cast(node);
    });

    p.def("getRoot", &Node::getRoot);
    p.def("getPath", &Node::getPathName);

    p.def("__getattr__", [](Node& self, const std::string& name) -> py::object
    {
        /// Custom properties.

        BaseObject *object = self.getObject(name);
        if (object)
            return py::cast(object);

        Node *child = self.getChild(name);
        if (child)
            return py::cast(child);

        return BindingBase::GetAttr(self, name);
    });

    p.def_property_readonly("children", [](Node* node)
    {
        return new NodeIterator(node,
                                [](Node* n) -> size_t { return n->child.size(); },
                                [](Node* n, size_t index) -> Node::SPtr { return n->child[index]; }
                                );
    });

    p.def_property_readonly("parents", [](Node* node)
    {
        return new NodeIterator(node,
                                [](Node* n) -> size_t { return n->getNbParents(); },
                                [](Node* n, size_t index) -> Node::SPtr {
                                        auto p = n->getParents();
                                        return (Node*)(p[index]);
                                }
                                );
    });

    p.def("__old_getChildren", [](Node& node)
    {
        py::list l;
        for(auto& child : node.child)
            l.append( py::cast(child) );
        return l;
    });

    p.def("__old_getChild", [](Node& node, size_t t)
    {
        if(t >= node.child.size())
            throw py::index_error("Index trop grand");
        return py::cast(node.child[t]);
    });

}
