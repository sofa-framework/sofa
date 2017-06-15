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
#ifndef SOFA_SIMULATION_INITACTION_H
#define SOFA_SIMULATION_INITACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/Visitor.h>

namespace sofa
{

namespace simulation
{

/** Initialize a newly created (or modified) scene graph.

    Forward: simulation::Node::initialize() This method puts the OdeSolver, if any, first in the list of components. Then BaseObject::init() for all components.

    Backward: OdeSolver::bwdInit()

    */
class SOFA_SIMULATION_CORE_API InitVisitor : public Visitor
{

public:
    InitVisitor(const core::ExecParams* params):Visitor(params),rootNode(NULL) {}


    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "init"; }
    virtual const char* getClassName() const { return "InitVisitor"; }

protected:
    Node *rootNode;
};

} // namespace simulation

} // namespace sofa

#endif
