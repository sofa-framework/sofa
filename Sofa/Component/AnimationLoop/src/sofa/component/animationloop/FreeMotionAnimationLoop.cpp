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
#include <sofa/component/animationloop/FreeMotionAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalVInitVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVInitVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalBeginIntegrationVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalComputeGeometricStiffness.h>
using sofa::simulation::mechanicalvisitor::MechanicalComputeGeometricStiffness;

#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalEndIntegrationVisitor;

namespace sofa::component::animationloop
{

using namespace core::behavior;
using namespace sofa::simulation;
using sofa::helper::ScopedAdvancedTimer;

using DefaultConstraintSolver = sofa::component::constraint::lagrangian::solver::GenericConstraintSolver;

FreeMotionAnimationLoop::FreeMotionAnimationLoop() :
    m_solveVelocityConstraintFirst(initData(&m_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
    , d_parallelCollisionDetectionAndFreeMotion(initData(&d_parallelCollisionDetectionAndFreeMotion, false, "parallelCollisionDetectionAndFreeMotion", "If true, executes free motion step and collision detection step in parallel."))
    , d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel during the free motion step."))
    , l_constraintSolver(initLink("constraintSolver", "The ConstraintSolver used in this animation loop (required)"))
{
    d_parallelCollisionDetectionAndFreeMotion.setGroup("Multithreading");
    d_parallelODESolving.setGroup("Multithreading");
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
= default;

void FreeMotionAnimationLoop::init()
{
    Inherit::init();

    simulation::common::VectorOperations vop(core::execparams::defaultInstance(), getContext());

    MultiVecDeriv dx(&vop, core::VecDerivId::dx());
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    MultiVecDeriv df(&vop, core::VecDerivId::dforce());
    df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    if (!l_constraintSolver)
    {
        l_constraintSolver.set(this->getContext()->get<sofa::core::behavior::ConstraintSolver>(core::objectmodel::BaseContext::SearchDown));
        if (!l_constraintSolver)
        {
            if (const auto constraintSolver = sofa::core::objectmodel::New<DefaultConstraintSolver>())
            {
                getContext()->addObject(constraintSolver);
                constraintSolver->setName( this->getContext()->getNameHelper().resolveName(constraintSolver->getClassName(), {}));
                constraintSolver->f_printLog.setValue(this->f_printLog.getValue());
                l_constraintSolver.set(constraintSolver);

                msg_warning() << "A ConstraintSolver is required by " << this->getClassName() << " but has not been found:"
                    " a default " << constraintSolver->getClassName() << " is automatically added in the scene for you. To remove this warning, add"
                    " a ConstraintSolver in the scene. The list of available constraint solvers is: "
                    << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::ConstraintSolver>();
            }
            else
            {
                msg_fatal() << "A ConstraintSolver is required by " << this->getClassName() << " but has not been found:"
                    " a default " << DefaultConstraintSolver::GetClass()->className << " could not be automatically added in the scene. To remove this error, add"
                    " a ConstraintSolver in the scene. The list of available constraint solvers is: "
                    << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::ConstraintSolver>();
                this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
        }
        else
        {
            msg_info() << "Constraint solver found: '" << l_constraintSolver->getPathName() << "'";
        }
    }

    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    if (d_parallelCollisionDetectionAndFreeMotion.getValue() || d_parallelODESolving.getValue())
    {
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

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    dmsg_info() << "################### step begin ###################";

    if (dt == 0)
        dt = node->getDt();
    
    double startTime = node->getTime();

    simulation::common::VectorOperations vop(params, node);
    simulation::common::MechanicalOperations mop(params, getContext());

    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecCoord freePos(&vop, core::VecCoordId::freePosition() );
    MultiVecDeriv freeVel(&vop, core::VecDerivId::freeVelocity() );

    core::ConstraintParams cparams(*params);
    cparams.setX(freePos);
    cparams.setV(freeVel);
    cparams.setDx(l_constraintSolver->getDx());
    cparams.setLambda(l_constraintSolver->getLambda());
    cparams.setOrder(m_solveVelocityConstraintFirst.getValue() ? core::ConstraintOrder::VEL : core::ConstraintOrder::POS_AND_VEL);

    MultiVecDeriv dx(&vop, core::VecDerivId::dx());
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    MultiVecDeriv df(&vop, core::VecDerivId::dforce());
    df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    {
        SCOPED_TIMER("MechanicalVInitVisitor");
        MechanicalVInitVisitor< core::V_COORD >(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(node);
        MechanicalVInitVisitor< core::V_DERIV >(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity(), true).execute(node);
    }

    // This animation loop works with lagrangian constraints. Forces derive from the constraints.
    // Therefore we notice the States that they have to consider them in the total accumulation of
    // forces.
    for (auto* state : node->getTreeObjects<sofa::core::BaseState>())
    {
        state->addToTotalForces(cparams.lambda().getId(state));
    }


#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        SCOPED_TIMER("AnimateBeginEvent");
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    dmsg_info() << "updatePos called" ;

    {
        SCOPED_TIMER("UpdatePosition");
        BehaviorUpdatePositionVisitor beh(params, dt);
        node->execute(&beh);
    }

    dmsg_info() << "updatePos performed - updateInternal called" ;

    UpdateInternalDataVisitor iud(params);

    dmsg_info() << "updateInternal called" ;

    {
        SCOPED_TIMER("updateInternalData");
        node->execute(&iud);
    }

    dmsg_info() << "updateInternal performed - beginVisitor called" ;


    // MechanicalBeginIntegrationVisitor
    MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    node->execute(&beginVisitor);

    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    // Mapping geometric stiffness coming from previous lambda.
    {
        SCOPED_TIMER("lambdaMultInvDt");
        MechanicalVOpVisitor lambdaMultInvDt(params, cparams.lambda(), sofa::core::ConstMultiVecId::null(), cparams.lambda(), 1.0 / dt);
        lambdaMultInvDt.setMapped(true);
        node->executeVisitor(&lambdaMultInvDt);
    }

    {
        SCOPED_TIMER("MechanicalComputeGeometricStiffness");
        MechanicalComputeGeometricStiffness geometricStiffnessVisitor(&mop.mparams, cparams.lambda());
        node->executeVisitor(&geometricStiffnessVisitor);
    }

    computeFreeMotionAndCollisionDetection(params, cparams, dt, pos, freePos, freeVel, &mop);

    // Solve constraints
    if (l_constraintSolver)
    {
        SCOPED_TIMER("ConstraintSolver");

        if (cparams.constOrder() == core::ConstraintOrder::VEL )
        {
            l_constraintSolver->solveConstraint(&cparams, vel);
            pos.eq(pos, vel, dt); //position += velocity * dt
        }
        else
        {
            l_constraintSolver->solveConstraint(&cparams, pos, vel);
        }

        MultiVecDeriv cdx(&vop, l_constraintSolver->getDx());
        mop.projectResponse(cdx);
        mop.propagateDx(cdx, true);
    }

    MechanicalEndIntegrationVisitor endVisitor(params, dt);
    node->execute(&endVisitor);

    mop.projectPositionAndVelocity(pos, vel);
    mop.propagateXAndV(pos, vel);
    
    node->setTime ( startTime + dt );
    node->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        SCOPED_TIMER("AnimateEndEvent");
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    {
        SCOPED_TIMER("UpdateMapping");
        //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
        node->execute<UpdateMappingVisitor>(params);
        {
            UpdateMappingEndEvent ev ( dt );
            PropagateEventVisitor act ( params , &ev );
            node->execute ( act );
        }
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

void FreeMotionAnimationLoop::computeFreeMotionAndCollisionDetection(const sofa::core::ExecParams* params,
                                                              const core::ConstraintParams& cparams, SReal dt,
                                                              sofa::core::MultiVecId pos,
                                                              sofa::core::MultiVecId freePos,
                                                              sofa::core::MultiVecDerivId freeVel,
                                                              simulation::common::MechanicalOperations* mop)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    if (!d_parallelCollisionDetectionAndFreeMotion.getValue())
    {
        SCOPED_TIMER("FreeMotion+CollisionDetection");

        computeFreeMotion(params, cparams, dt, pos, freePos, freeVel, mop);

        {
            ScopedAdvancedTimer collisionDetectionTimer("CollisionDetection");
            computeCollision(params);
        }
    }
    else
    {
        SCOPED_TIMER("FreeMotion+CollisionDetection");

        auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
        assert(taskScheduler != nullptr);

        preCollisionComputation(params);

        {
            ScopedAdvancedTimer collisionResetTimer("CollisionReset");
            CollisionResetVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        sofa::simulation::CpuTask::Status freeMotionTaskStatus;
        taskScheduler->addTask(freeMotionTaskStatus, [&]() { computeFreeMotion(params, cparams, dt, pos, freePos, freeVel, mop); });

        {
            ScopedAdvancedTimer collisionDetectionTimer("CollisionDetection");
            CollisionDetectionVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        {
            ScopedAdvancedTimer waitFreeMotionTimer("WaitFreeMotion");
            taskScheduler->workUntilDone(&freeMotionTaskStatus);
        }

        {
            ScopedAdvancedTimer collisionResponseTimer("CollisionResponse");
            CollisionResponseVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        postCollisionComputation(params);
    }
}

void FreeMotionAnimationLoop::computeFreeMotion(const sofa::core::ExecParams* params, const core::ConstraintParams& cparams, SReal dt,
                                         sofa::core::MultiVecId pos,
                                         sofa::core::MultiVecId freePos,
                                         sofa::core::MultiVecDerivId freeVel,
                                         simulation::common::MechanicalOperations* mop)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    {
        SCOPED_TIMER("FreeMotion");
        simulation::SolveVisitor freeMotion(params, dt, true, d_parallelODESolving.getValue());
        node->execute(&freeMotion);
    }

    mop->projectResponse(freeVel);
    mop->propagateDx(freeVel, true);

    if (cparams.constOrder() == sofa::core::ConstraintOrder::POS ||
        cparams.constOrder() == sofa::core::ConstraintOrder::POS_AND_VEL)
    {
        SCOPED_TIMER("freePosEqPosPlusFreeVelDt");
        MechanicalVOpVisitor freePosEqPosPlusFreeVelDt(params, freePos, pos, freeVel, dt);
        freePosEqPosPlusFreeVelDt.setMapped(true);
        node->executeVisitor(&freePosEqPosPlusFreeVelDt);
    }
}

int FreeMotionAnimationLoopClass = core::RegisterObject(R"(
The animation loop to use with constraints.
You must add this loop at the beginning of the scene if you are using constraints.")")
                                   .add< FreeMotionAnimationLoop >()
                                   .addAlias("FreeMotionMasterSolver");

} //namespace sofa::component::animationloop
