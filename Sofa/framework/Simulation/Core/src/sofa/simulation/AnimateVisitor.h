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
#pragma once

#include <sofa/simulation/config.h>
#include <sofa/simulation/fwd.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/behavior/fwd.h>

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API AnimateVisitor : public Visitor
{

protected :
    SReal dt;
    bool firstNodeVisited;

public:
    AnimateVisitor(const core::ExecParams* params, SReal dt);

    void setDt(SReal v) { dt = v; }
    SReal getDt() const { return dt; }

    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);
    virtual void fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* obj);

    Result processNodeTopDown(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "animate"; }
    const char* getClassName() const override { return "AnimateVisitor"; }
};

} // namespace sofa::simulation
