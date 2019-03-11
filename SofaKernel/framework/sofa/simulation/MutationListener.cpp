/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

void MutationListener::onStepBegin(Node *node) { SOFA_UNUSED(node); }
void MutationListener::onStepEnd(Node *node) { SOFA_UNUSED(node); }

void MutationListener::sleepChanged(Node *node) { SOFA_UNUSED(node); }

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

} // namespace simulation

} // namespace sofa
