/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_COMMON_MUTATIONLISTENER_H
#define SOFA_SIMULATION_COMMON_MUTATIONLISTENER_H

#include <sofa/SofaSimulation.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace simulation
{

class Node;

///Generic API to handle mutations of the graph
class SOFA_SIMULATION_COMMON_API MutationListener
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
};

} // namespace simulation

} // namespace sofa

#endif
