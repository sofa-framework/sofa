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
#ifndef SOFA_SIMULATION_CORE_MUTATIONLISTENER_H
#define SOFA_SIMULATION_CORE_MUTATIONLISTENER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/simulationcore.h>

namespace sofa
{

namespace simulation
{

class Node;

/// Generic API to handle mutations of the graph
class SOFA_SIMULATION_CORE_API MutationListener
{
  public:
    virtual ~MutationListener();

    virtual void addChildBegin(Node *parent, Node *child) final;

    virtual void removeChildBegin(Node *parent, Node *child) final;

    virtual void addObjectBegin(Node *parent,
                                core::objectmodel::BaseObject *object) final;

    virtual void removeObjectBegin(Node *parent,
                                   core::objectmodel::BaseObject *object) final;

    virtual void addSlaveBegin(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave) final;

    virtual void removeSlaveBegin(core::objectmodel::BaseObject *master,
                                  core::objectmodel::BaseObject *slave) final;

    virtual void sleepChanged(Node *node);

    virtual void addChildEnd(Node *parent, Node *child) final;

    virtual void removeChildEnd(Node *parent, Node *child) final;

    virtual void addObjectEnd(Node *parent,
                              core::objectmodel::BaseObject *object) final;

    virtual void removeObjectEnd(Node *parent,
                                 core::objectmodel::BaseObject *object) final;

    virtual void addSlaveEnd(core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave) final;

    virtual void removeSlaveEnd(core::objectmodel::BaseObject *master,
                                core::objectmodel::BaseObject *slave) final;

  protected:
    virtual void doAddChildBegin(Node *parent, Node *child);

    virtual void doRemoveChildBegin(Node *parent, Node *child);

    virtual void doAddObjectBegin(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void doRemoveObjectBegin(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void doMoveObjectBegin(Node *previous, Node *parent,
                              core::objectmodel::BaseObject *object);

    virtual void doAddSlaveBegin(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void doRemoveSlaveBegin(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);

    virtual void doMoveSlaveBegin(core::objectmodel::BaseObject *previousMaster,
                             core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave);

    virtual void doAddChildEnd(Node *parent, Node *child);

    virtual void doRemoveChildEnd(Node *parent, Node *child);

    virtual void doAddObjectEnd(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void doRemoveObjectEnd(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void doMoveObjectEnd(Node *previous, Node *parent,
                              core::objectmodel::BaseObject *object);

    virtual void doAddSlaveEnd(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void doRemoveSlaveEnd(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);

    virtual void doMoveSlaveEnd(core::objectmodel::BaseObject *previousMaster,
                             core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave);
};

} // namespace simulation

} // namespace sofa

#endif
