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

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateMatrixDeriv.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>


namespace sofa::simulation
{

void registerDefaultAnimationLoop(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Simulation loop, created by default when the user does not define one in the scene. This loop first computes the collision detection and then solves the physics.")
        .add<DefaultAnimationLoop>()
        .addDocumentationURL(std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("components/animationloops/defaultanimationloop/"))
        .addDescription(R"(
This loop triggers the following steps:
- build and solve all linear systems in the scene : collision and time integration to compute the new values of the dofs
- update the context (dt++)
- update the mappings
- update the bounding box (volume covering all objects of the scene))"));
}

DefaultAnimationLoop::DefaultAnimationLoop(simulation::Node* _m_node)
    : Inherit()
    , d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel"))
{
    SOFA_UNUSED(_m_node);
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
    Inherit::init();
    if (!l_node)
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
    l_node.set(n);
}

void DefaultAnimationLoop::behaviorUpdatePosition(const core::ExecParams* params, const SReal dt) const
{
    SCOPED_TIMER("BehaviorUpdatePositionVisitor");
    BehaviorUpdatePositionVisitor beh(params, dt);
    m_node->execute(beh);
}

void DefaultAnimationLoop::updateInternalData(const core::ExecParams* params) const
{
    SCOPED_TIMER("UpdateInternalDataVisitor");
    m_node->execute<UpdateInternalDataVisitor>(params);
}

void DefaultAnimationLoop::updateSimulationContext(const core::ExecParams* params, const SReal dt, const SReal startTime) const
{
    SCOPED_TIMER("UpdateSimulationContextVisitor");
    m_node->setTime(startTime + dt);
    m_node->execute<UpdateSimulationContextVisitor>(params);
}

void DefaultAnimationLoop::propagateAnimateEndEvent(const core::ExecParams* params, const SReal dt) const
{
    AnimateEndEvent ev(dt);
    PropagateEventVisitor propagateEventVisitor(params, &ev);
    m_node->execute(propagateEventVisitor);
}

void DefaultAnimationLoop::updateMapping(const core::ExecParams* params, const SReal dt) const
{
    SCOPED_TIMER("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    m_node->execute<UpdateMappingVisitor>(params);
    {
        UpdateMappingEndEvent ev(dt);
        PropagateEventVisitor propagateEventVisitor(params, &ev);
        m_node->execute(propagateEventVisitor);
    }
}

void DefaultAnimationLoop::computeBoundingBox(const core::ExecParams* params) const
{
    if (d_computeBoundingBox.getValue())
    {
        SCOPED_TIMER("UpdateBBox");
        m_node->execute<UpdateBoundingBoxVisitor>(params);
    }
}

void DefaultAnimationLoop::propagateAnimateBeginEvent(const core::ExecParams* params, const SReal dt) const
{
    AnimateBeginEvent ev(dt);
    PropagateEventVisitor act(params, &ev);
    m_node->execute(act);
}

void DefaultAnimationLoop::beginIntegration(const core::ExecParams* params, SReal dt) const
{
    propagateIntegrateBeginEvent(params);

    SCOPED_TIMER("beginIntegration");
    mechanicalvisitor::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    m_node->execute(&beginVisitor);
}

void DefaultAnimationLoop::propagateIntegrateBeginEvent(const core::ExecParams* params) const
{
    SCOPED_TIMER("propagateIntegrateBeginEvent");
    IntegrateBeginEvent evBegin;
    PropagateEventVisitor eventPropagation( params, &evBegin);
    eventPropagation.execute(m_node);
}

void DefaultAnimationLoop::accumulateMatrixDeriv(const core::ConstraintParams cparams) const
{
    SCOPED_TIMER("accumulateMatrixDeriv");
    mechanicalvisitor::MechanicalAccumulateMatrixDeriv accumulateMatrixDeriv(&cparams, core::vec_id::write_access::constraintJacobian);
    accumulateMatrixDeriv.execute(m_node);
}

void DefaultAnimationLoop::solve(const core::ExecParams* params, SReal dt) const
{
    constexpr bool usefreeVecIds = false;
    constexpr bool computeForceIsolatedInteractionForceFields = true;
    SCOPED_TIMER("solve");
    simulation::SolveVisitor freeMotion(params, dt, usefreeVecIds, d_parallelODESolving.getValue(), computeForceIsolatedInteractionForceFields);
    freeMotion.execute(m_node);
}

void DefaultAnimationLoop::propagateIntegrateEndEvent(const core::ExecParams* params) const
{
    SCOPED_TIMER("propagateIntegrateEndEvent");
    IntegrateEndEvent evBegin;
    PropagateEventVisitor eventPropagation(params, &evBegin);
    eventPropagation.execute(m_node);
}

void DefaultAnimationLoop::endIntegration(const core::ExecParams* params, const SReal dt) const
{
    {
        SCOPED_TIMER("endIntegration");
        mechanicalvisitor::MechanicalEndIntegrationVisitor endVisitor(params, dt);
        m_node->execute(&endVisitor);
    }

    propagateIntegrateEndEvent(params);
}

void DefaultAnimationLoop::projectPositionAndVelocity(const SReal nextTime, const sofa::core::MechanicalParams& mparams) const
{
    SCOPED_TIMER("projectPositionAndVelocity");
    mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor(&mparams, nextTime,
                                                                   sofa::core::vec_id::write_access::position, sofa::core::vec_id::write_access::velocity
    ).execute( m_node );
}

void DefaultAnimationLoop::propagateOnlyPositionAndVelocity(const SReal nextTime, const sofa::core::MechanicalParams& mparams) const
{
    SCOPED_TIMER("propagateOnlyPositionAndVelocity");
    mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, nextTime,
                                                                         core::vec_id::write_access::position,
                                                                         core::vec_id::write_access::velocity).execute( m_node );
}

void DefaultAnimationLoop::propagateCollisionBeginEvent(const core::ExecParams* params) const
{
    SCOPED_TIMER("CollisionBeginEvent");
    CollisionBeginEvent evBegin;
    PropagateEventVisitor eventPropagation( params, &evBegin);
    eventPropagation.execute(m_node);
}

void DefaultAnimationLoop::propagateCollisionEndEvent(const core::ExecParams* params) const
{
    SCOPED_TIMER("CollisionEndEvent");
    CollisionEndEvent evEnd;
    PropagateEventVisitor eventPropagation( params, &evEnd);
    eventPropagation.execute(m_node);
}

void DefaultAnimationLoop::collisionDetection(const core::ExecParams* params) const
{
    propagateCollisionBeginEvent(params);

    {
        SCOPED_TIMER("collision");
        CollisionVisitor act(params);
        m_node->execute(&act);
    }

    propagateCollisionEndEvent(params);
}

void DefaultAnimationLoop::animate(const core::ExecParams* params, SReal dt) const
{
    const SReal startTime = m_node->getTime();
    const SReal nextTime = startTime + dt;

    sofa::core::MechanicalParams mparams(*params);
    mparams.setDt(dt);

    behaviorUpdatePosition(params, dt);
    updateInternalData(params);

    collisionDetection(params);

    beginIntegration(params, dt);
    {
        const core::ConstraintParams cparams;
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

    m_node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());
    assert(m_node);

    if (dt == 0_sreal)
    {
        dt = m_node->getDt();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    propagateAnimateBeginEvent(params, dt);
    animate(params, dt);
    updateSimulationContext(params, dt, m_node->getTime());
    propagateAnimateEndEvent(params, dt);

    updateMapping(params, dt);
    computeBoundingBox(params);

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}


} // namespace sofa
