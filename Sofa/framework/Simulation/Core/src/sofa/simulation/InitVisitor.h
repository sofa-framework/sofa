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
#ifndef SOFA_SIMULATION_INITACTION_H
#define SOFA_SIMULATION_INITACTION_H



#include <sofa/simulation/Visitor.h>


namespace sofa::simulation
{

/** Initialize a newly created (or modified) scene graph.

    Forward: simulation::Node::initialize() This method puts the OdeSolver, if any, first in the list of components. Then BaseObject::init() for all components.

    Backward: OdeSolver::bwdInit()

    */
class SOFA_SIMULATION_CORE_API InitVisitor : public Visitor
{

public:
    InitVisitor(const core::ExecParams* eparams):Visitor(eparams),rootNode(nullptr) {}


    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "init"; }
    const char* getClassName() const override { return "InitVisitor"; }

protected:
    Node *rootNode;
};

} // namespace sofa::simulation


#endif
