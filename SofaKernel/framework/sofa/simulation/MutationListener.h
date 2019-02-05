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

#include <sofa/simulation/simulationcore.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace simulation
{

class Node;

///Generic API to handle mutations of the graph
class SOFA_SIMULATION_CORE_API MutationListener
{
public:
    virtual ~MutationListener();

    virtual void addChild(Node* parent, Node* child) final;

    virtual void removeChild(Node* parent, Node* child) final;

    virtual void moveChild(Node* previous, Node* parent, Node* child) final;

    virtual void addObject(Node* parent, core::objectmodel::BaseObject* object) final;

    virtual void removeObject(Node* parent, core::objectmodel::BaseObject* object) final;

    virtual void moveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object) final;

    virtual void addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) final;

    virtual void removeSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) final;

    virtual void moveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) final;

    virtual void sleepChanged(Node* node);


    virtual void addChildDone(Node* parent, Node* child);

    virtual void removeChildDone(Node* parent, Node* child);

    virtual void moveChildDone(Node* previous, Node* parent, Node* child);

    virtual void addObjectDone(Node* parent, core::objectmodel::BaseObject* object);

    virtual void removeObjectDone(Node* parent, core::objectmodel::BaseObject* object);

    virtual void moveObjectDone(Node* previous, Node* parent, core::objectmodel::BaseObject* object);


protected:
    virtual void doAddChild(Node* parent, Node* child);

    virtual void doRemoveChild(Node* parent, Node* child);

    virtual void doMoveChild(Node* previous, Node* parent, Node* child);

    virtual void doAddObject(Node* parent, core::objectmodel::BaseObject* object);

    virtual void doRemoveObject(Node* parent, core::objectmodel::BaseObject* object);

    virtual void doMoveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object);

    virtual void doAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

    virtual void doRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

    virtual void doMoveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);
};

} // namespace simulation

} // namespace sofa

#endif
