/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

    virtual void addChild(Node* parent, Node* child);

    virtual void removeChild(Node* parent, Node* child);

    virtual void moveChild(Node* previous, Node* parent, Node* child);

    virtual void addObject(Node* parent, core::objectmodel::BaseObject* object);

    virtual void removeObject(Node* parent, core::objectmodel::BaseObject* object);

    virtual void moveObject(Node* previous, Node* parent, core::objectmodel::BaseObject* object);

    virtual void addSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

    virtual void removeSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

    virtual void moveSlave(core::objectmodel::BaseObject* previousMaster, core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave);

	virtual void sleepChanged(Node* node);
};

} // namespace simulation

} // namespace sofa

#endif
