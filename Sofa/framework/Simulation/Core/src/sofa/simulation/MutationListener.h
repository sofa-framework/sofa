/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_SIMULATION_CORE_MUTATIONLISTENER_H
#define SOFA_SIMULATION_CORE_MUTATIONLISTENER_H

#include <sofa/simulation/config.h>
#include <sofa/simulation/fwd.h>
#include <sofa/core/fwd.h>


namespace sofa::simulation
{

class Node;

/// Generic API to handle mutations of the graph
class SOFA_SIMULATION_CORE_API MutationListener
{
  public:
    virtual ~MutationListener();

    virtual void sleepChanged(Node *node);

    virtual void onBeginAddChild(Node *parent, Node *child);

    virtual void onBeginRemoveChild(Node *parent, Node *child);

    virtual void onBeginAddObject(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void onBeginRemoveObject(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void onBeginAddSlave(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void onBeginRemoveSlave(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);

    virtual void onEndAddChild(Node *parent, Node *child);

    virtual void onEndRemoveChild(Node *parent, Node *child);

    virtual void onEndAddObject(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void onEndRemoveObject(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void onEndAddSlave(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void onEndRemoveSlave(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);
};

} // namespace sofa::simulation


#endif
