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

void MutationListener::sleepChanged(Node *node) { SOFA_UNUSED(node); }

void MutationListener::onBeginAddChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onBeginRemoveChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}

void MutationListener::onBeginAddObject(Node *parent,
                                      core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onBeginRemoveObject(Node *parent,
                                         core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onBeginAddSlave(core::objectmodel::BaseObject *master,
                                     core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onBeginRemoveSlave(core::objectmodel::BaseObject *master,
                                        core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

void MutationListener::onEndAddChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onEndRemoveChild(Node *parent, Node *child)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(child);
}
void MutationListener::onEndAddObject(Node *parent,
                                    core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onEndRemoveObject(Node *parent,
                                       core::objectmodel::BaseObject *object)
{
    SOFA_UNUSED(parent);
    SOFA_UNUSED(object);
}
void MutationListener::onEndAddSlave(core::objectmodel::BaseObject *master,
                                   core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}
void MutationListener::onEndRemoveSlave(core::objectmodel::BaseObject *master,
                                      core::objectmodel::BaseObject *slave)
{
    SOFA_UNUSED(master);
    SOFA_UNUSED(slave);
}

} // namespace simulation

} // namespace sofa
