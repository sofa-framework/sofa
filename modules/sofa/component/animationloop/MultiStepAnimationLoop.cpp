/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/animationloop/MultiStepAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <math.h>
#include <iostream>




namespace sofa
{

namespace component
{

namespace animationloop
{

int MultiStepAnimationLoopClass = core::RegisterObject("Multi steps animation loop, multi integration steps in a single animation step are managed.")
        .add< MultiStepAnimationLoop >()
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

void MultiStepAnimationLoop::step(const sofa::core::ExecParams* params /* PARAMS FIRST */, double dt)
{
    sofa::helper::AdvancedTimer::stepBegin("AnimationStep");

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    double startTime = this->gnode->getTime();

    BehaviorUpdatePositionVisitor beh(params , dt);
    this->gnode->execute ( beh );

    const int ncollis = collisionSteps.getValue();
    const int ninteg = integrationSteps.getValue();
    double stepDt = dt / (ncollis * ninteg);
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
            sout << "integration" << sendl;
            integrate(params, stepDt);
        }
    }

    this->gnode->setTime ( startTime + dt );
    this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

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
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif

    sofa::helper::AdvancedTimer::stepEnd("AnimationStep");
}

} // namespace animationloop

} // namespace component

} // namespace sofa

