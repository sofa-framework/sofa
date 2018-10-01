#ifndef PYTHONMODULE_SOFA_BINDING_NODE_H
#define PYTHONMODULE_SOFA_BINDING_NODE_H

#include <SofaPython3/Sofa/Core/Binding_BaseObject.h>

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

template class py::class_<Base, Base::SPtr>;
template class py::class_<sofa::core::objectmodel::BaseNode, Base,
                          sofa::core::objectmodel::BaseNode::SPtr>;

template class py::class_<sofa::core::objectmodel::BaseContext,
                          sofa::core::objectmodel::Base,
                          sofa::core::objectmodel::BaseContext::SPtr>;

template class py::class_<sofa::core::objectmodel::Context,
                          sofa::core::objectmodel::BaseContext,
                          sofa::core::objectmodel::Context::SPtr>;

template class py::class_<Node, sofa::core::objectmodel::BaseNode,
                          sofa::core::objectmodel::Context, Node::SPtr>;

void moduleAddNode(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_NODE_H
