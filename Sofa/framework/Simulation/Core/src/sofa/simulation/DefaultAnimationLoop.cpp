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
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateMatrixDeriv.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBuildConstraintMatrix.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>


namespace sofa::simulation
{

int DefaultAnimationLoopClass = core::RegisterObject("Simulation loop to use in scene without constraints nor contact.")
                                .add<DefaultAnimationLoop>()
                                .addDescription(R"(
This loop triggers the following steps:
- build and solve all linear systems in the scene : collision and time integration to compute the new values of the dofs
- update the context (dt++)
- update the mappings
- update the bounding box (volume covering all objects of the scene))");

DefaultAnimationLoop::DefaultAnimationLoop(simulation::Node* _gnode)
    : Inherit()
    , d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel"))
    , gnode(_gnode)
{
    //assert(gnode);

    this->addUpdateCallback("parallelODESolving", {&d_parallelODESolving},
    [this](const core::DataTracker& tracker) -> sofa::core::objectmodel::ComponentState
    {
        SOFA_UNUSED(tracker);
        if (d_parallelODESolving.getValue())
        {
            simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
            assert(taskScheduler);

            if (taskScheduler->getThreadCount() < 1)
            {
                taskScheduler->init(0);
                msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
            }
            else
            {
                msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
            }
        }
        return d_componentState.getValue();
    },
{});
}

DefaultAnimationLoop::~DefaultAnimationLoop() = default;

void DefaultAnimationLoop::init()
{
    if (!gnode)
    {
        gnode = dynamic_cast<simulation::Node*>(this->getContext());
    }

    if (!gnode)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

void DefaultAnimationLoop::setNode(simulation::Node* n)
{
    gnode = n;
}

void DefaultAnimationLoop::behaviorUpdatePosition(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("BehaviorUpdatePositionVisitor");
    BehaviorUpdatePositionVisitor beh(params, dt);
    gnode->execute(beh);
}

void DefaultAnimationLoop::updateInternalData(const core::ExecParams* params) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateInternalDataVisitor");
    gnode->execute<UpdateInternalDataVisitor>(params);
}

void DefaultAnimationLoop::updateSimulationContext(const core::ExecParams* params, const SReal dt, const SReal startTime) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateSimulationContextVisitor");
    gnode->setTime(startTime + dt);
    gnode->execute<UpdateSimulationContextVisitor>(params);
}

void DefaultAnimationLoop::propagateAnimateEndEvent(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("propagateAnimateEndEvent");
    AnimateEndEvent ev(dt);
    PropagateEventVisitor propagateEventVisitor(params, &ev);
    gnode->execute(propagateEventVisitor);
}

void DefaultAnimationLoop::updateMapping(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    gnode->execute<UpdateMappingVisitor>(params);
    {
        UpdateMappingEndEvent ev(dt);
        PropagateEventVisitor propagateEventVisitor(params, &ev);
        gnode->execute(propagateEventVisitor);
    }
}

void DefaultAnimationLoop::computeBoundingBox(const core::ExecParams* params) const
{
    if (d_computeBoundingBox.getValue())
    {
        sofa::helper::ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute<UpdateBoundingBoxVisitor>(params);
    }
}

void DefaultAnimationLoop::propagateAnimateBeginEvent(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("propagateAnimateBeginEvent");
    AnimateBeginEvent ev(dt);
    PropagateEventVisitor act(params, &ev);
    gnode->execute(act);
}

void DefaultAnimationLoop::resetConstraint(const core::ExecParams* params) const
{
    sofa::helper::ScopedAdvancedTimer timer("resetConstraint");
    const sofa::core::ConstraintParams cparams(*params);
    sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor resetConstraint(&cparams);
    gnode->execute(&resetConstraint);
}

void DefaultAnimationLoop::beginIntegration(const core::ExecParams* params, SReal dt) const
{
    propagateIntegrateBeginEvent(params);

    sofa::helper::ScopedAdvancedTimer timer("beginIntegration");
    mechanicalvisitor::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    gnode->execute(&beginVisitor);
}

void DefaultAnimationLoop::propagateIntegrateBeginEvent(const core::ExecParams* params) const
{
    sofa::helper::ScopedAdvancedTimer timer("propagateIntegrateBeginEvent");
    IntegrateBeginEvent evBegin;
    PropagateEventVisitor eventPropagation( params, &evBegin);
    eventPropagation.execute(gnode);
}

void DefaultAnimationLoop::buildConstraintMatrix(core::ConstraintParams cparams) const
{
    sofa::helper::ScopedAdvancedTimer timer("buildConstraintMatrix");
    unsigned int constraintId = 0;
    mechanicalvisitor::MechanicalBuildConstraintMatrix buildConstraintMatrix(&cparams, core::MatrixDerivId::constraintJacobian(), constraintId );
    buildConstraintMatrix.execute(gnode);
}

void DefaultAnimationLoop::accumulateMatrixDeriv(const core::ConstraintParams cparams) const
{
    sofa::helper::ScopedAdvancedTimer timer("accumulateMatrixDeriv");
    mechanicalvisitor::MechanicalAccumulateMatrixDeriv accumulateMatrixDeriv(&cparams, core::MatrixDerivId::constraintJacobian());
    accumulateMatrixDeriv.execute(gnode);
}

void DefaultAnimationLoop::solve(const core::ExecParams* params, SReal dt) const
{
    constexpr bool usefreeVecIds = false;
    constexpr bool computeForceIsolatedInteractionForceFields = true;
    sofa::helper::ScopedAdvancedTimer timer("solve");
    simulation::SolveVisitor freeMotion(params, dt, usefreeVecIds, d_parallelODESolving.getValue(), computeForceIsolatedInteractionForceFields);
    freeMotion.execute(gnode);
}

void DefaultAnimationLoop::propagateIntegrateEndEvent(const core::ExecParams* params) const
{
    sofa::helper::ScopedAdvancedTimer timer("propagateIntegrateEndEvent");
    IntegrateEndEvent evBegin;
    PropagateEventVisitor eventPropagation(params, &evBegin);
    eventPropagation.execute(gnode);
}

void DefaultAnimationLoop::endIntegration(const core::ExecParams* params, const SReal dt) const
{
    {
        sofa::helper::ScopedAdvancedTimer timer("endIntegration");
        mechanicalvisitor::MechanicalEndIntegrationVisitor endVisitor(params, dt);
        gnode->execute(&endVisitor);
    }

    propagateIntegrateEndEvent(params);
}

void DefaultAnimationLoop::projectPositionAndVelocity(const SReal nextTime, const sofa::core::MechanicalParams& mparams) const
{
    sofa::helper::ScopedAdvancedTimer timer("projectPositionAndVelocity");
    mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor(&mparams, nextTime,
                                                                   sofa::core::VecCoordId::position(), sofa::core::VecDerivId::velocity()
    ).execute( gnode );
}

void DefaultAnimationLoop::propagateOnlyPositionAndVelocity(const SReal nextTime, const sofa::core::MechanicalParams& mparams) const
{
    sofa::helper::ScopedAdvancedTimer timer("propagateOnlyPositionAndVelocity");
    mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, nextTime,
                                                                         core::VecCoordId::position(),
                                                                         core::VecDerivId::velocity()).execute( gnode );
}

void DefaultAnimationLoop::propagateCollisionBeginEvent(const core::ExecParams* params) const
{
    CollisionBeginEvent evBegin;
    PropagateEventVisitor eventPropagation( params, &evBegin);
    eventPropagation.execute(gnode);
}

void DefaultAnimationLoop::propagateCollisionEndEvent(const core::ExecParams* params) const
{
    CollisionEndEvent evEnd;
    PropagateEventVisitor eventPropagation( params, &evEnd);
    eventPropagation.execute(gnode);
}

void DefaultAnimationLoop::collisionDetection(const core::ExecParams* params) const
{
    propagateCollisionBeginEvent(params);

    {
        sofa::helper::ScopedAdvancedTimer timer("collision");
        CollisionVisitor act(params);
        gnode->execute(&act);
    }

    propagateCollisionEndEvent(params);
}

void DefaultAnimationLoop::animate(const core::ExecParams* params, SReal dt) const
{
    const SReal startTime = gnode->getTime();
    const SReal nextTime = startTime + dt;

    sofa::core::MechanicalParams mparams(*params);
    mparams.setDt(dt);

    behaviorUpdatePosition(params, dt);
    updateInternalData(params);

    resetConstraint(params);

    collisionDetection(params);

    beginIntegration(params, dt);
    {
        const core::ConstraintParams cparams;
        buildConstraintMatrix(cparams);
        accumulateMatrixDeriv(cparams);

        solve(params, dt);

        projectPositionAndVelocity(nextTime, mparams);
        propagateOnlyPositionAndVelocity(nextTime, mparams);
    }
    endIntegration(params, dt);
}

void DefaultAnimationLoop::step(const core::ExecParams* params, SReal dt)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
    {
        return;
    }

    if (dt == 0_sreal)
    {
        dt = this->gnode->getDt();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    propagateAnimateBeginEvent(params, dt);
    animate(params, dt);
    updateSimulationContext(params, dt, gnode->getTime());
    propagateAnimateEndEvent(params, dt);

    updateMapping(params, dt);
    computeBoundingBox(params);

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}


} // namespace sofa
