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


MutationListener::~MutationListener()
{
}

void MutationListener::addChild(Node* parent, Node* child)
{
    doAddChild(parent, child);
    for(Node::ObjectIterator it = child->object.begin(); it != child->object.end(); ++it)
        addObject(child, it->get());
    for(Node::ChildIterator it = child->child.begin(); it != child->child.end(); ++it)
        addChild(child, it->get());
}

void MutationListener::removeChild(Node* parent, Node* child)
{
    doRemoveChild(parent, child);
    for(Node::ObjectIterator it = child->object.begin(); it != child->object.end(); ++it)
        removeObject(child, it->get());
    for(Node::ChildIterator it = child->child.begin(); it != child->child.end(); ++it)
        removeChild(child, it->get());
}

void MutationListener::addObject(Node* parent, core::objectmodel::BaseObject* object)
{
    doAddObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves& slaves = object->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        addSlave(object, slaves[i].get());
}

void MutationListener::removeObject(Node* parent, core::objectmodel::BaseObject* object)
{
    doRemoveObject(parent, object);
    const core::objectmodel::BaseObject::VecSlaves& slaves = object->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        removeSlave(object, slaves[i].get());
}

void MutationListener::moveChild(Node* previous, Node* parent, Node* child)
{
    doMoveChild(previous, parent, child);
    removeChild(previous, child);
    addChild(parent, child);
}

void MutationListener::moveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object)
{
    doMoveObject(previous, parent, object);
    removeObject(previous, object);
    addObject(parent, object);
}

void MutationListener::addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    doAddSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves& slaves = slave->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        addSlave(slave, slaves[i].get());
}

void MutationListener::removeSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    doRemoveSlave(master, slave);
    const core::objectmodel::BaseObject::VecSlaves& slaves = slave->getSlaves();
    for (unsigned int i=0; i<slaves.size(); ++i)
        removeSlave(slave, slaves[i].get());
}

void MutationListener::moveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave)
{
    doMoveSlave(previousMaster, master, slave);
    removeSlave(previousMaster, slave);
    addSlave(master, slave);
}

void MutationListener::sleepChanged(Node* node)
{
    SOFA_UNUSED(node);
}

void MutationListener::doAddChild(Node* parent, Node* child) { SOFA_UNUSED(parent); SOFA_UNUSED(child); }

void MutationListener::doRemoveChild(Node* parent, Node* child) { SOFA_UNUSED(parent); SOFA_UNUSED(child); }

void MutationListener::doMoveChild(Node* previous, Node* parent, Node* child) { SOFA_UNUSED(parent); SOFA_UNUSED(previous); SOFA_UNUSED(child); }

void MutationListener::doAddObject(Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(parent); SOFA_UNUSED(object); }

void MutationListener::doRemoveObject(Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(parent); SOFA_UNUSED(object); }

void MutationListener::doMoveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(previous); SOFA_UNUSED(parent); SOFA_UNUSED(object); }

void MutationListener::doAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) { SOFA_UNUSED(master); SOFA_UNUSED(slave); }

void MutationListener::doRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) { SOFA_UNUSED(master); SOFA_UNUSED(slave); }

void MutationListener::doMoveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) { SOFA_UNUSED(previousMaster); SOFA_UNUSED(master); SOFA_UNUSED(slave); }


void MutationListener::addChildDone(Node* parent, Node* child) { SOFA_UNUSED(parent); SOFA_UNUSED(child); }

void MutationListener::removeChildDone(Node* parent, Node* child) { SOFA_UNUSED(parent); SOFA_UNUSED(child); }

void MutationListener::moveChildDone(Node* previous, Node* parent, Node* child) { SOFA_UNUSED(previous);  SOFA_UNUSED(parent); SOFA_UNUSED(child); }

void MutationListener::addObjectDone(Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(parent); SOFA_UNUSED(object); }

void MutationListener::removeObjectDone(Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(parent); SOFA_UNUSED(object); }

void MutationListener::moveObjectDone(Node* previous, Node* parent, core::objectmodel::BaseObject* object) { SOFA_UNUSED(previous); SOFA_UNUSED(parent); SOFA_UNUSED(object); }


} // namespace simulation

} // namespace sofa

