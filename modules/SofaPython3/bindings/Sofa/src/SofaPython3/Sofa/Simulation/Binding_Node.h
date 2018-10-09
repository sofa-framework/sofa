#ifndef PYTHONMODULE_SOFA_BINDING_NODE_H
#define PYTHONMODULE_SOFA_BINDING_NODE_H
#include <functional>
#include <pybind11/pybind11.h>
#include <SofaPython3/Sofa/Core/Binding_BaseObject.h>

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

template class pybind11::class_<sofa::core::objectmodel::BaseNode,
                                sofa::core::objectmodel::Base,
                                sofa::core::objectmodel::BaseNode::SPtr>;

template class pybind11::class_<sofa::core::objectmodel::BaseContext,
                                sofa::core::objectmodel::Base,
                                sofa::core::objectmodel::BaseContext::SPtr>;

template class pybind11::class_<sofa::core::objectmodel::Context,
                                sofa::core::objectmodel::BaseContext,
                                sofa::core::objectmodel::Context::SPtr>;

template class pybind11::class_<Node,
                                sofa::core::objectmodel::BaseNode,
                                sofa::core::objectmodel::Context,
                                Node::SPtr>;


namespace sofapython3
{
namespace py { using namespace pybind11; }

class BaseIterator
{
public:
    Node::SPtr owner;
    size_t     index=0;
    std::function<size_t (Node*)> size ;
    std::function<Base::SPtr (Node*, size_t)> get ;

    BaseIterator(Node::SPtr owner_,
                 std::function<size_t (Node*)> size_,
                 std::function<Base::SPtr (Node*, size_t)> get_)
    {
        size = size_;
        get = get_;
        owner=owner_;
        index=0;
    }
};



void moduleAddNode(py::module &m);

} /// namespace sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_NODE_H
