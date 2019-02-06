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

    virtual void moveChildBegin(Node *previous, Node *parent,
                                Node *child) final;

    virtual void addObjectBegin(Node *parent,
                                core::objectmodel::BaseObject *object) final;

    virtual void removeObjectBegin(Node *parent,
                                   core::objectmodel::BaseObject *object) final;

    virtual void moveObjectBegin(Node *previous, Node *parent,
                                 core::objectmodel::BaseObject *object) final;

    virtual void addSlaveBegin(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave) final;

    virtual void removeSlaveBegin(core::objectmodel::BaseObject *master,
                                  core::objectmodel::BaseObject *slave) final;

    virtual void moveSlaveBegin(core::objectmodel::BaseObject *previousMaster,
                                core::objectmodel::BaseObject *master,
                                core::objectmodel::BaseObject *slave) final;

    virtual void sleepChanged(Node *node);

    virtual void addChildEnd(Node *parent, Node *child) final;

    virtual void removeChildEnd(Node *parent, Node *child) final;

    virtual void moveChildEnd(Node *previous, Node *parent, Node *child) final;

    virtual void addObjectEnd(Node *parent,
                              core::objectmodel::BaseObject *object) final;

    virtual void removeObjectEnd(Node *parent,
                                 core::objectmodel::BaseObject *object) final;

    virtual void moveObjectEnd(Node *previous, Node *parent,
                               core::objectmodel::BaseObject *object) final;

    virtual void addSlaveEnd(core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave) final;

    virtual void removeSlaveEnd(core::objectmodel::BaseObject *master,
                                core::objectmodel::BaseObject *slave) final;

    virtual void moveSlaveEnd(core::objectmodel::BaseObject *previousMaster,
                              core::objectmodel::BaseObject *master,
                              core::objectmodel::BaseObject *slave) final;

  protected:
    virtual void beginAddChild(Node *parent, Node *child);

    virtual void beginRemoveChild(Node *parent, Node *child);

    virtual void beginMoveChild(Node *previous, Node *parent, Node *child);

    virtual void beginAddObject(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void beginRemoveObject(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void beginMoveObject(Node *previous, Node *parent,
                              core::objectmodel::BaseObject *object);

    virtual void beginAddSlave(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void beginRemoveSlave(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);

    virtual void beginMoveSlave(core::objectmodel::BaseObject *previousMaster,
                             core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave);

    virtual void endAddChild(Node *parent, Node *child);

    virtual void endRemoveChild(Node *parent, Node *child);

    virtual void endMoveChild(Node *previous, Node *parent, Node *child);

    virtual void endAddObject(Node *parent,
                             core::objectmodel::BaseObject *object);

    virtual void endRemoveObject(Node *parent,
                                core::objectmodel::BaseObject *object);

    virtual void endMoveObject(Node *previous, Node *parent,
                              core::objectmodel::BaseObject *object);

    virtual void endAddSlave(core::objectmodel::BaseObject *master,
                            core::objectmodel::BaseObject *slave);

    virtual void endRemoveSlave(core::objectmodel::BaseObject *master,
                               core::objectmodel::BaseObject *slave);

    virtual void endMoveSlave(core::objectmodel::BaseObject *previousMaster,
                             core::objectmodel::BaseObject *master,
                             core::objectmodel::BaseObject *slave);
};

} // namespace simulation

} // namespace sofa

#endif
