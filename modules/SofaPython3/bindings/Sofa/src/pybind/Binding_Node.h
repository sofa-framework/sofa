#ifndef PYTHONMODULE_SOFA_BINDING_NODE_H
#define PYTHONMODULE_SOFA_BINDING_NODE_H

#include "Binding_BaseObject.h"

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

template class py::class_<Base, Base::SPtr>;
template class py::class_<Node, Base, Node::SPtr>;

void moduleAddNode(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_NODE_H
