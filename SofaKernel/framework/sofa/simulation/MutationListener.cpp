/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, development version     *
 *                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation; either version 2.1 of the License, or (at     *
 * your option) any later version.                                             *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
 * for more details.                                                           *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.        *
 *******************************************************************************
 * Authors: The SOFA Team and external contributors (see Authors.txt)          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include <sofa/simulation/Node.h>

#include <sofa/simulation/MutationListener.h>

namespace sofa
{

namespace simulation
{

MutationListener::~MutationListener() {}

void MutationListener::addChildBegin(Node *parent, Node *child)
{
    onAddChildBegin(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        addObjectBegin(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        addChildBegin(child, it->get());
}

void MutationListener::removeChildBegin(Node *parent, Node *child)
{
    // Removing a child is like detaching a child. it doesn't mean detaching
    // the whole descendency from that child, just detaching it from its parent
    onRemoveChildBegin(parent, child);
}

void MutationListener::addObjectBegin(Node *parent,
                                      core::objectmodel::BaseObject *object)
{
    onAddObjectBegin(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(object, slaves[i].get());
}

void MutationListener::removeObjectBegin(Node *parent,
                                         core::objectmodel::BaseObject *object)
{
    // Removing an object is like detaching an object. it doesn't mean detaching
    // the whole descendency from that object, just detaching it from its parent
    onRemoveObjectBegin(parent, object);
}


void MutationListener::addSlaveBegin(core::objectmodel::BaseObject *master,
                                     core::objectmodel::BaseObject *slave)
{
    onAddSlaveBegin(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(slave, slaves[i].get());
}

void MutationListener::removeSlaveBegin(core::objectmodel::BaseObject *master,
                                        core::objectmodel::BaseObject *slave)
{
    // Whatever slaves are...
    onRemoveSlaveBegin(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveBegin(slave, slaves[i].get());
}

void MutationListener::sleepChanged(Node *node) { SOFA_UNUSED(node); }

void MutationListener::addChildEnd(Node *parent, Node *child)
{
    onAddChildEnd(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        addObjectEnd(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        addChildEnd(child, it->get());
}

void MutationListener::removeChildEnd(Node *parent, Node *child)
{
    // Removing a child is like detaching a child. it doesn't mean detaching
    // the whole descendency from that child, just detaching it from its parent
    onRemoveChildEnd(parent, child);
}

void MutationListener::addObjectEnd(Node *parent,
                                    core::objectmodel::BaseObject *object)
{
    onAddObjectEnd(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveEnd(object, slaves[i].get());
}

void MutationListener::removeObjectEnd(Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    // Removing an object is like detaching an object. it doesn't mean detaching
    // the whole descendency from that object, just detaching it from its parent
    onRemoveObjectEnd(parent, object);
}

void MutationListener::addSlaveEnd(core::objectmodel::BaseObject *master,
                                   core::objectmodel::BaseObject *slave)
{
    onAddSlaveEnd(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(slave, slaves[i].get());
}

void MutationListener::removeSlaveEnd(core::objectmodel::BaseObject *master,
                                      core::objectmodel::BaseObject *slave)
{
    onRemoveSlaveEnd(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveEnd(slave, slaves[i].get());
}

void MutationListener::onAddChildBegin(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onRemoveChildBegin(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onAddObjectBegin(Node *parent,
                                      core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onRemoveObjectBegin(Node *parent,
                                         core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onMoveObjectBegin(Node *previous, Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onAddSlaveBegin(core::objectmodel::BaseObject *master,
                                     core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onRemoveSlaveBegin(core::objectmodel::BaseObject *master,
                                        core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onMoveSlaveBegin(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(previousMaster);
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

void MutationListener::onAddChildEnd(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onRemoveChildEnd(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onAddObjectEnd(Node *parent,
                                    core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onRemoveObjectEnd(Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onMoveObjectEnd(Node *previous, Node *parent,
                                     core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onAddSlaveEnd(core::objectmodel::BaseObject *master,
                                   core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onRemoveSlaveEnd(core::objectmodel::BaseObject *master,
                                      core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onMoveSlaveEnd(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(previousMaster);
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

} // namespace simulation

} // namespace sofa
