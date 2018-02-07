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
#ifndef SOFA_SIMULATION_ANIMATEACTION_H
#define SOFA_SIMULATION_ANIMATEACTION_H

#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/collision/Pipeline.h>

namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_CORE_API AnimateVisitor : public Visitor
{

protected :
    SReal dt;
    bool firstNodeVisited;
public:
    AnimateVisitor(const core::ExecParams* params = core::ExecParams::defaultInstance());
    AnimateVisitor(const core::ExecParams* params, SReal dt);

    void setDt(SReal v) { dt = v; }
    SReal getDt() const { return dt; }

    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);
    virtual void processBehaviorModel(simulation::Node* node, core::BehaviorModel* obj);
    virtual void fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* obj);
    virtual void processOdeSolver(simulation::Node* node, core::behavior::OdeSolver* obj);

    virtual Result processNodeTopDown(simulation::Node* node);
    //virtual void processNodeBottomUp(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "animate"; }
    virtual const char* getClassName() const { return "AnimateVisitor"; }
};

} // namespace simulation

} // namespace sofa

#endif
