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
    beginAddChild(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        addObjectBegin(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        addChildBegin(child, it->get());
}

void MutationListener::removeChildBegin(Node *parent, Node *child)
{
    beginRemoveChild(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        removeObjectBegin(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        removeChildBegin(child, it->get());
}

void MutationListener::moveChildBegin(Node *previous, Node *parent, Node *child)
{
    beginMoveChild(previous, parent, child);
}

void MutationListener::addObjectBegin(Node *parent,
                                      core::objectmodel::BaseObject *object)
{
    beginAddObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(object, slaves[i].get());
}

void MutationListener::removeObjectBegin(Node *parent,
                                         core::objectmodel::BaseObject *object)
{
    beginRemoveObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveBegin(object, slaves[i].get());
}

void MutationListener::moveObjectBegin(Node *previous, Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    beginMoveObject(previous, parent, object);
    removeObjectBegin(previous, object);
    addObjectBegin(parent, object);
}

void MutationListener::addSlaveBegin(core::objectmodel::BaseObject *master,
                                     core::objectmodel::BaseObject *slave)
{
    beginAddSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(slave, slaves[i].get());
}

void MutationListener::removeSlaveBegin(core::objectmodel::BaseObject *master,
                                        core::objectmodel::BaseObject *slave)
{
    beginRemoveSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveBegin(slave, slaves[i].get());
}

void MutationListener::moveSlaveBegin(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    beginMoveSlave(previousMaster, master, slave);
    removeSlaveBegin(previousMaster, slave);
    addSlaveBegin(master, slave);
}

void MutationListener::sleepChanged(Node *node) { SOFA_UNUSED(node); }

void MutationListener::addChildEnd(Node *parent, Node *child)
{
    endAddChild(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        addObjectEnd(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        addChildEnd(child, it->get());
}

void MutationListener::removeChildEnd(Node *parent, Node *child)
{
    endRemoveChild(parent, child);
    for (Node::ObjectIterator it = child->object.begin();
         it != child->object.end(); ++it)
        removeObjectEnd(child, it->get());
    for (Node::ChildIterator it = child->child.begin();
         it != child->child.end(); ++it)
        removeChildEnd(child, it->get());
}

void MutationListener::moveChildEnd(Node *previous, Node *parent, Node *child)
{
    endMoveChild(previous, parent, child);
    removeChildBegin(previous, child);
    addChildBegin(parent, child);
}

void MutationListener::addObjectEnd(Node *parent,
                                    core::objectmodel::BaseObject *object)
{
    endAddObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveEnd(object, slaves[i].get());
}

void MutationListener::removeObjectEnd(Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    endRemoveObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves &slaves =
        object->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveEnd(object, slaves[i].get());
}

void MutationListener::moveObjectEnd(Node *previous, Node *parent,
                                     core::objectmodel::BaseObject *object)
{
    endMoveObject(previous, parent, object);
    removeObjectEnd(previous, object);
    addObjectEnd(parent, object);
}

void MutationListener::addSlaveEnd(core::objectmodel::BaseObject *master,
                                   core::objectmodel::BaseObject *slave)
{
    endAddSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        addSlaveBegin(slave, slaves[i].get());
}

void MutationListener::removeSlaveEnd(core::objectmodel::BaseObject *master,
                                      core::objectmodel::BaseObject *slave)
{
    endRemoveSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves &slaves = slave->getSlaves();
    for (unsigned int i = 0; i < slaves.size(); ++i)
        removeSlaveEnd(slave, slaves[i].get());
}

void MutationListener::moveSlaveEnd(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    endMoveSlave(previousMaster, master, slave);
    removeSlaveEnd(previousMaster, slave);
    addSlaveEnd(master, slave);
}

void MutationListener::beginAddChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::beginRemoveChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::beginMoveChild(Node *previous, Node *parent, Node *child)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::beginAddObject(Node *parent,
                                      core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::beginRemoveObject(Node *parent,
                                         core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::beginMoveObject(Node *previous, Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::beginAddSlave(core::objectmodel::BaseObject *master,
                                     core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::beginRemoveSlave(core::objectmodel::BaseObject *master,
                                        core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::beginMoveSlave(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(previousMaster);
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

void MutationListener::endAddChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::endRemoveChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::endMoveChild(Node *previous, Node *parent, Node *child)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::endAddObject(Node *parent,
                                    core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::endRemoveObject(Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::endMoveObject(Node *previous, Node *parent,
                                     core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(previous);
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::endAddSlave(core::objectmodel::BaseObject *master,
                                   core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::endRemoveSlave(core::objectmodel::BaseObject *master,
                                      core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::endMoveSlave(
    core::objectmodel::BaseObject *previousMaster,
    core::objectmodel::BaseObject *master, core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(previousMaster);
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

} // namespace simulation

} // namespace sofa
