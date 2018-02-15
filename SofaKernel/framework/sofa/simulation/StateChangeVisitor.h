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
#ifndef SOFA_SIMULATION_TREE_STATECHANGEACTION_H
#define SOFA_SIMULATION_TREE_STATECHANGEACTION_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/topology/Topology.h>

namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_CORE_API StateChangeVisitor : public Visitor
{

public:
    StateChangeVisitor(const sofa::core::ExecParams* params, core::topology::Topology* source);

    virtual void processStateChange(core::behavior::BaseMechanicalState* obj);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "stateChange"; }
    virtual const char* getClassName() const { return "StateChangeVisitor"; }

protected:
    bool root;
    core::topology::Topology* source;
};


} // namespace simulation

} // namespace sofa

#endif
