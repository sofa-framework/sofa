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
#include <SofaGeneralAnimationLoop/MultiStepAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <math.h>
#include <iostream>


using namespace sofa::simulation;

namespace sofa
{

namespace component
{

namespace animationloop
{

int MultiStepAnimationLoopClass = core::RegisterObject("Multi steps animation loop, multi integration steps in a single animation step are managed.")
        .add< MultiStepAnimationLoop >()
        .addAlias("MultiStepMasterSolver")
        ;

SOFA_DECL_CLASS(MultiStepAnimationLoop);

MultiStepAnimationLoop::MultiStepAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , collisionSteps( initData(&collisionSteps,1,"collisionSteps", "number of collision steps between each frame rendering") )
    , integrationSteps( initData(&integrationSteps,1,"integrationSteps", "number of integration steps between each collision detection") )
{
}

MultiStepAnimationLoop::~MultiStepAnimationLoop()
{
}

void MultiStepAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();


    sofa::helper::AdvancedTimer::stepBegin("AnimationStep");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    SReal startTime = this->gnode->getTime();

    BehaviorUpdatePositionVisitor beh(params , dt);
    this->gnode->execute ( beh );

    const int ncollis = collisionSteps.getValue();
    const int ninteg = integrationSteps.getValue();

    SReal stepDt = dt / (ncollis * ninteg);
    for (int c = 0; c < ncollis; ++c)
    {
        // First we reset the constraints
        sofa::simulation::MechanicalResetConstraintVisitor(params).execute(this->getContext());
        // Then do collision detection and response creation
        sout << "collision" << sendl;
        computeCollision(params);
        for (int i = 0; i < ninteg; ++i)
        {
            // Then integrate the time step
            sout << "integration at time = " << startTime+i*stepDt << sendl;
            integrate(params, stepDt);

            this->gnode->setTime ( startTime + (i+1)*stepDt );
            this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        }
    }


    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        this->gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

#ifndef SOFA_NO_UPDATE_BBOX
    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    this->gnode->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif

    sofa::helper::AdvancedTimer::stepEnd("AnimationStep");
}

} // namespace animationloop

} // namespace component

} // namespace sofa

