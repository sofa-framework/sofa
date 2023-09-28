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
#include <sofa/component/animationloop/MultiStepAnimationLoop.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;

using namespace sofa::simulation;

namespace sofa::component::animationloop
{

int MultiStepAnimationLoopClass = core::RegisterObject("Multi steps animation loop, multi integration steps in a single animation step are managed.")
        .add< MultiStepAnimationLoop >()
        .addAlias("MultiStepMasterSolver")
        ;

MultiStepAnimationLoop::MultiStepAnimationLoop() :
      collisionSteps( initData(&collisionSteps,1,"collisionSteps", "number of collision steps between each frame rendering") )
    , integrationSteps( initData(&integrationSteps,1,"integrationSteps", "number of integration steps between each collision detection") )
{
}

MultiStepAnimationLoop::~MultiStepAnimationLoop()
{
}

void MultiStepAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    if (dt == 0)
        dt = node->getDt();

    SCOPED_TIMER_VARNAME(animationStepTimer, "AnimationStep");

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }
    
    SReal startTime = node->getTime();

    BehaviorUpdatePositionVisitor beh(params , dt);
    node->execute ( beh );

    UpdateInternalDataVisitor uid(params);
    node->execute ( uid );

    const int ncollis = collisionSteps.getValue();
    const int ninteg = integrationSteps.getValue();

    SReal stepDt = dt / (ncollis * ninteg);

    // initialize a constraint params object with default MultiVecId for 
    // constraint jacobian, free positions, free velocity vectors
    sofa::core::ConstraintParams cparams(*params); 

    std::stringstream tmpStr;
    for (int c = 0; c < ncollis; ++c)
    {
        // First we reset the constraints
        MechanicalResetConstraintVisitor(&cparams).execute(node);
        // Then do collision detection and response creation
        tmpStr << "collision" ;

        computeCollision(params);
        for (int i = 0; i < ninteg; ++i)
        {
            // Then integrate the time step
            tmpStr << "integration at time = " << startTime+i*stepDt << msgendl;
            integrate(params, stepDt);
            
            node->setTime ( startTime + (i+1)*stepDt );
            node->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        }
    }
    msg_info() << tmpStr.str();
    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    {
        SCOPED_TIMER_VARNAME(updateMappingTimer, "UpdateMapping");
        node->execute<UpdateMappingVisitor>(params);
    }
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        node->execute ( act );
    }

    if (d_computeBoundingBox.getValue())
    {
        SCOPED_TIMER("UpdateBBox");
        node->execute<UpdateBoundingBoxVisitor>(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}

} // namespace sofa::component::animationloop
