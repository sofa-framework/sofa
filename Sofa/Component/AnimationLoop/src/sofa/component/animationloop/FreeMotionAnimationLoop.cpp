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

#include <sofa/component/constraint/lagrangian/solver/LCPConstraintSolver.h>

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

FreeMotionAnimationLoop::FreeMotionAnimationLoop(simulation::Node* gnode)
    : Inherit1(gnode)
    , m_solveVelocityConstraintFirst(initData(&m_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
    , d_parallelCollisionDetectionAndFreeMotion(initData(&d_parallelCollisionDetectionAndFreeMotion, false, "parallelCollisionDetectionAndFreeMotion", "If true, executes free motion step and collision detection step in parallel."))
    , d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel during the free motion step."))
    , defaultSolver(nullptr)
    , l_constraintSolver(initLink("constraintSolver", "The ConstraintSolver used in this animation loop (required)"))
{
    d_parallelCollisionDetectionAndFreeMotion.setGroup("Multithreading");
    d_parallelODESolving.setGroup("Multithreading");
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
{
    if (defaultSolver != nullptr)
        defaultSolver.reset();
}

void FreeMotionAnimationLoop::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    simulation::CollisionAnimationLoop::parse(arg);

    defaultSolver = sofa::core::objectmodel::New<constraint::lagrangian::solver::LCPConstraintSolver>();
    defaultSolver->parse(arg);
    defaultSolver->setName(defaultSolver->getContext()->getNameHelper().resolveName(defaultSolver->getClassName(), core::ComponentNameHelper::Convention::python));
}


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
            if (defaultSolver != nullptr)
            {
                msg_warning() << "A ConstraintSolver is required by " << this->getClassName() << " but has not been found:"
                    " a default " << defaultSolver->getClassName() << " is automatically added in the scene for you. To remove this warning, add"
                    " a ConstraintSolver in the scene. The list of available constraint solvers is: "
                    << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::ConstraintSolver>();
                getContext()->addObject(defaultSolver);
                l_constraintSolver.set(defaultSolver);
                defaultSolver = nullptr;
            }
            else
            {
                msg_fatal() << "A ConstraintSolver is required by " << this->getClassName() << " but has not been found:"
                    " a default LCPConstraintSolver could not be automatically added in the scene. To remove this error, add"
                    " a ConstraintSolver in the scene. The list of available constraint solvers is: "
                    << core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::ConstraintSolver>();
            }
        }
        else
        {
            msg_info() << "Constraint solver found: '" << l_constraintSolver->getPathName() << "'";
        }
    }
    else
    {
        defaultSolver.reset();
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
}


void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    dmsg_info() << "################### step begin ###################";

    if (dt == 0)
        dt = gnode->getDt();

    double startTime = gnode->getTime();

    simulation::common::VectorOperations vop(params, getContext());
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
    cparams.setOrder(m_solveVelocityConstraintFirst.getValue() ? core::ConstraintParams::VEL : core::ConstraintParams::POS_AND_VEL);

    MultiVecDeriv dx(&vop, core::VecDerivId::dx());
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    MultiVecDeriv df(&vop, core::VecDerivId::dforce());
    df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    {
        ScopedAdvancedTimer timer("MechanicalVInitVisitor");
        MechanicalVInitVisitor< core::V_COORD >(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(gnode);
        MechanicalVInitVisitor< core::V_DERIV >(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity(), true).execute(gnode);
    }


#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        ScopedAdvancedTimer timer("AnimateBeginEvent");
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    dmsg_info() << "updatePos called" ;

    {
        ScopedAdvancedTimer timer("UpdatePosition");
        BehaviorUpdatePositionVisitor beh(params, dt);
        gnode->execute(&beh);
    }

    dmsg_info() << "updatePos performed - updateInternal called" ;

    UpdateInternalDataVisitor iud(params);

    dmsg_info() << "updateInternal called" ;

    {
        ScopedAdvancedTimer timer("updateInternalData");
        gnode->execute(&iud);
    }

    dmsg_info() << "updateInternal performed - beginVisitor called" ;


    // MechanicalBeginIntegrationVisitor
    MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    gnode->execute(&beginVisitor);

    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    // Mapping geometric stiffness coming from previous lambda.
    {
        ScopedAdvancedTimer timer("lambdaMultInvDt");
        MechanicalVOpVisitor lambdaMultInvDt(params, cparams.lambda(), sofa::core::ConstMultiVecId::null(), cparams.lambda(), 1.0 / dt);
        lambdaMultInvDt.setMapped(true);
        getContext()->executeVisitor(&lambdaMultInvDt);
    }

    {
        ScopedAdvancedTimer timer("MechanicalComputeGeometricStiffness");
        MechanicalComputeGeometricStiffness geometricStiffnessVisitor(&mop.mparams, cparams.lambda());
        getContext()->executeVisitor(&geometricStiffnessVisitor);
    }

    FreeMotionAndCollisionDetection(params, cparams, dt, pos, freePos, freeVel, &mop);

    // Solve constraints
    if (l_constraintSolver)
    {
        ScopedAdvancedTimer timer("ConstraintSolver");

        if (cparams.constOrder() == core::ConstraintParams::VEL )
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
    gnode->execute(&endVisitor);

    mop.projectPositionAndVelocity(pos, vel);
    mop.propagateXAndV(pos, vel);

    gnode->setTime ( startTime + dt );
    gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        ScopedAdvancedTimer timer("AnimateEndEvent");
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }

    {
        ScopedAdvancedTimer timer("UpdateMapping");
        //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
        gnode->execute<UpdateMappingVisitor>(params);
        {
            UpdateMappingEndEvent ev ( dt );
            PropagateEventVisitor act ( params , &ev );
            gnode->execute ( act );
        }
    }

    if (d_computeBoundingBox.getValue())
    {
        ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute<UpdateBoundingBoxVisitor>(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif

}

void FreeMotionAnimationLoop::FreeMotionAndCollisionDetection(const sofa::core::ExecParams* params,
                                                              const core::ConstraintParams& cparams, SReal dt,
                                                              sofa::core::MultiVecId pos,
                                                              sofa::core::MultiVecId freePos,
                                                              sofa::core::MultiVecDerivId freeVel,
                                                              simulation::common::MechanicalOperations* mop)
{
    if (!d_parallelCollisionDetectionAndFreeMotion.getValue())
    {
        ScopedAdvancedTimer timer("FreeMotion+CollisionDetection");

        computeFreeMotion(params, cparams, dt, pos, freePos, freeVel, mop);

        {
            ScopedAdvancedTimer collisionDetectionTimer("CollisionDetection");
            computeCollision(params);
        }
    }
    else
    {
        ScopedAdvancedTimer timer("FreeMotion+CollisionDetection");

        auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
        assert(taskScheduler != nullptr);

        preCollisionComputation(params);

        {
            ScopedAdvancedTimer collisionResetTimer("CollisionReset");
            CollisionResetVisitor act(params);
            act.setTags(this->getTags());
            act.execute(getContext());
        }

        sofa::simulation::CpuTask::Status freeMotionTaskStatus;
        taskScheduler->addTask(freeMotionTaskStatus, [&]() { computeFreeMotion(params, cparams, dt, pos, freePos, freeVel, mop); });

        {
            ScopedAdvancedTimer collisionDetectionTimer("CollisionDetection");
            CollisionDetectionVisitor act(params);
            act.setTags(this->getTags());
            act.execute(getContext());
        }

        {
            ScopedAdvancedTimer waitFreeMotionTimer("WaitFreeMotion");
            taskScheduler->workUntilDone(&freeMotionTaskStatus);
        }

        {
            ScopedAdvancedTimer collisionResponseTimer("CollisionResponse");
            CollisionResponseVisitor act(params);
            act.setTags(this->getTags());
            act.execute(getContext());
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
    {
        sofa::helper::ScopedAdvancedTimer timer("FreeMotion");
        simulation::SolveVisitor freeMotion(params, dt, true, d_parallelODESolving.getValue());
        gnode->execute(&freeMotion);
    }

    mop->projectResponse(freeVel);
    mop->propagateDx(freeVel, true);

    if (cparams.constOrder() == sofa::core::ConstraintParams::POS ||
        cparams.constOrder() == sofa::core::ConstraintParams::POS_AND_VEL)
    {
        sofa::helper::ScopedAdvancedTimer timer("freePosEqPosPlusFreeVelDt");
        MechanicalVOpVisitor freePosEqPosPlusFreeVelDt(params, freePos, pos, freeVel, dt);
        freePosEqPosPlusFreeVelDt.setMapped(true);
        getContext()->executeVisitor(&freePosEqPosPlusFreeVelDt);
    }
}

int FreeMotionAnimationLoopClass = core::RegisterObject(R"(
The animation loop to use with constraints.
You must add this loop at the beginning of the scene if you are using constraints.")")
                                   .add< FreeMotionAnimationLoop >()
                                   .addAlias("FreeMotionMasterSolver");

} //namespace sofa::component::animationloop
