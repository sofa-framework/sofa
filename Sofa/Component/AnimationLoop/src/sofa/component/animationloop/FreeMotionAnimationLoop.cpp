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
#include <sofa/simulation/events/AnimateBeginEvent.h>
#include <sofa/simulation/events/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/events/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/component/constraint/lagrangian/solver/BlockGaussSeidelConstraintSolver.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalVInitVisitor.h>

#include <sofa/component/constraint/lagrangian/solver/BlockGaussSeidelConstraintSolver.h>
using sofa::simulation::mechanicalvisitor::MechanicalVInitVisitor;

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

using DefaultConstraintSolver = sofa::component::constraint::lagrangian::solver::BlockGaussSeidelConstraintSolver;

FreeMotionAnimationLoop::FreeMotionAnimationLoop()
    : d_solveConstraints(initData(&d_solveConstraints, true, "solveConstraints", "If true, constraints are solved. Otherwise, they are ignored."))
    , d_constraintsStateVariables(initData(&d_constraintsStateVariables , ConstraintsStateVariables("posVel"), "constraintsStateVariables", "State variables on which the constraints are expressed and corrected: position only (statics), position and velocity, or velocity only"))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
    , d_parallelCollisionDetectionAndFreeMotion(initData(&d_parallelCollisionDetectionAndFreeMotion, false, "parallelCollisionDetectionAndFreeMotion", "If true, executes free motion step and collision detection step in parallel."))
    , d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel during the free motion step."))
    , l_constraintSolver(initLink("constraintSolver", "The ConstraintSolver used in this animation loop (required)"))
{
    d_parallelCollisionDetectionAndFreeMotion.setGroup("Multithreading");
    d_parallelODESolving.setGroup("Multithreading");

    this->addUpdateCallback("validateConstraintSolver", {&d_solveConstraints}, [this](const core::DataTracker& )
    {
        if (d_solveConstraints.getValue())
        {
            validateConstraintSolver();
            if (l_constraintSolver)
            {
                l_constraintSolver->init();
            }
        }
        return this->getComponentState();
    }, {});
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
= default;

void FreeMotionAnimationLoop::init()
{
    Inherit::init();

    simulation::common::VectorOperations vop(core::execparams::defaultInstance(), getContext());

    MultiVecDeriv dx(&vop, core::vec_id::write_access::dx);
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    MultiVecDeriv df(&vop, core::vec_id::write_access::dforce);
    df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    if (d_solveConstraints.getValue() && !l_constraintSolver)
    {
        validateConstraintSolver();
    }

    if (d_parallelCollisionDetectionAndFreeMotion.getValue() || d_parallelODESolving.getValue())
    {
        initTaskScheduler();
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void FreeMotionAnimationLoop::parse(BaseObjectDescription* arg)
{
    CollisionAnimationLoop::parse(arg);

    if (arg->getAttribute("solveVelocityConstraintFirst") && !d_constraintsStateVariables.isSet())
    {
        const auto value = arg->getAttributeAsInt("solveVelocityConstraintFirst", 0);
        if (value == 0)
        {
            d_constraintsStateVariables.setValue(ConstraintsStateVariables("posVel"));
        }
        else
        {
            d_constraintsStateVariables.setValue(ConstraintsStateVariables("vel"));
        }
    }
}

void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    if (isComponentStateInvalid())
        return;

    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    dmsg_info() << "################### step begin ###################";

    if (dt == 0)
        dt = node->getDt();
    
    double startTime = node->getTime();

    simulation::common::VectorOperations vop(params, node);
    simulation::common::MechanicalOperations mop(params, getContext());

    MultiVecCoord pos(&vop, core::vec_id::write_access::position );
    MultiVecDeriv vel(&vop, core::vec_id::write_access::velocity );

    const auto solveConstraints = d_solveConstraints.getValue();
    MultiVecCoord freePos(&vop, solveConstraints ? core::vec_id::write_access::freePosition : core::vec_id::write_access::position);
    MultiVecDeriv freeVel(&vop, solveConstraints ? core::vec_id::write_access::freeVelocity : core::vec_id::write_access::velocity );

    core::ConstraintParams cparams(*params);
    cparams.setX(freePos);
    cparams.setV(freeVel);
    if (l_constraintSolver)
    {
        cparams.setDx(l_constraintSolver->getDx());
        cparams.setLambda(l_constraintSolver->getLambda());
    }
    switch (d_constraintsStateVariables.getValue())
    {
        case ConstraintsStateVariables("pos"):
            cparams.setOrder(core::ConstraintOrder::POS);
            break;
        case ConstraintsStateVariables("posVel"):
            cparams.setOrder(core::ConstraintOrder::POS_AND_VEL);
            break;
        case ConstraintsStateVariables("vel"):
            cparams.setOrder(core::ConstraintOrder::VEL);
            break;
    }

    MultiVecDeriv dx(&vop, core::vec_id::write_access::dx);
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    MultiVecDeriv df(&vop, core::vec_id::write_access::dforce);
    df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    if (solveConstraints)
    {
        SCOPED_TIMER("MechanicalVInitVisitor");
        MechanicalVInitVisitor< core::V_COORD >(params, core::vec_id::write_access::freePosition, core::vec_id::read_access::position, true).execute(node);
        MechanicalVInitVisitor< core::V_DERIV >(params, core::vec_id::write_access::freeVelocity, core::vec_id::read_access::velocity, true).execute(node);

        MechanicalVOpVisitor(params, core::vec_id::write_access::freePosition, core::vec_id::read_access::position).setMapped(true).execute(node);
        MechanicalVOpVisitor(params, core::vec_id::write_access::freeVelocity, core::vec_id::read_access::velocity).setMapped(true).execute(node);
    }

    // This animation loop works with lagrangian constraints. Forces derive from the constraints.
    // Therefore we notice the States that they have to consider them in the total accumulation of
    // forces.
    if (solveConstraints)
    {
        for (auto* state : node->getTreeObjects<sofa::core::BaseState>())
        {
            state->addToTotalForces(cparams.lambda().getId(state));
        }
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

    if (solveConstraints)
    {
        SCOPED_TIMER("MechanicalComputeGeometricStiffness");
        MechanicalComputeGeometricStiffness geometricStiffnessVisitor(&mop.mparams, cparams.lambda());
        node->executeVisitor(&geometricStiffnessVisitor);
    }

    computeFreeMotionAndCollisionDetection(params, cparams, dt, pos, freePos, freeVel, &mop);

    // Solve constraints
    if (solveConstraints && l_constraintSolver)
    {
        SCOPED_TIMER("ConstraintSolver");

        const auto constOrder = cparams.constOrder();
        if (constOrder == core::ConstraintOrder::VEL )
        {
            l_constraintSolver->solveConstraint(&cparams, vel);
            pos.eq(pos, vel, dt); //position += velocity * dt
        }
        else if (constOrder == core::ConstraintOrder::POS_AND_VEL)
        {
            l_constraintSolver->solveConstraint(&cparams, pos, vel);
        }
        else if (constOrder == core::ConstraintOrder::POS)
        {
            l_constraintSolver->solveConstraint(&cparams, pos);
        }

        {
            SCOPED_TIMER("ProjectAndPropagateDx");

            MultiVecDeriv cdx(&vop, l_constraintSolver->getDx());
            mop.projectResponse(cdx);
            mop.propagateDx(cdx, true);
        }
    }

    MechanicalEndIntegrationVisitor endVisitor(params, dt);
    node->execute(&endVisitor);

    if (cparams.constOrder() == core::ConstraintOrder::POS)
    {
        SCOPED_TIMER("ProjectAndPropagateX");

        mop.projectPosition(pos);
        mop.propagateX(pos);
    }
    else
    {
        SCOPED_TIMER("ProjectAndPropagateXAndV");

        mop.projectPositionAndVelocity(pos, vel);
        mop.propagateXAndV(pos, vel);
    }

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
                                                              sofa::core::MultiVecCoordId pos,
                                                              sofa::core::MultiVecCoordId freePos,
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

        preCollisionComputation(params);

        {
            SCOPED_TIMER("ProcessGeometricalData");
            ProcessGeometricalDataVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        {
            SCOPED_TIMER("CollisionReset");
            CollisionResetVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        sofa::simulation::CpuTask::Status freeMotionTaskStatus;
        m_taskScheduler->addTask(freeMotionTaskStatus, [&]() { computeFreeMotion(params, cparams, dt, pos, freePos, freeVel, mop); });

        {
            SCOPED_TIMER("CollisionDetection");
            CollisionDetectionVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        {
            SCOPED_TIMER("WaitFreeMotion");
            m_taskScheduler->workUntilDone(&freeMotionTaskStatus);
        }

        {
            SCOPED_TIMER("CollisionResponse");
            CollisionResponseVisitor act(params);
            act.setTags(this->getTags());
            act.execute(node);
        }

        postCollisionComputation(params);
    }
}

void FreeMotionAnimationLoop::computeFreeMotion(const sofa::core::ExecParams* params, const core::ConstraintParams& cparams, SReal dt,
                                         sofa::core::MultiVecCoordId pos,
                                         sofa::core::MultiVecCoordId freePos,
                                         sofa::core::MultiVecDerivId freeVel,
                                         simulation::common::MechanicalOperations* mop)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    {
        SCOPED_TIMER("FreeMotion");
        simulation::SolveVisitor freeMotion(params, dt, d_solveConstraints.getValue(), d_parallelODESolving.getValue());
        node->execute(&freeMotion);
    }

    if (cparams.constOrder() == core::ConstraintOrder::POS)
    {
        mop->projectPosition(freePos);
        mop->propagateX(freePos);
    }
    else
    {
        mop->projectResponse(freeVel);
        mop->propagateDx(freeVel, true);

        if (cparams.constOrder() == sofa::core::ConstraintOrder::POS_AND_VEL && d_solveConstraints.getValue())
        {
            SCOPED_TIMER("freePosEqPosPlusFreeVelDt");
            MechanicalVOpVisitor freePosEqPosPlusFreeVelDt(params, freePos, pos, freeVel, dt);
            freePosEqPosPlusFreeVelDt.setMapped(true);
            node->executeVisitor(&freePosEqPosPlusFreeVelDt);
        }
    }
}

void FreeMotionAnimationLoop::validateConstraintSolver()
{
    l_constraintSolver.set(this->getContext()->get<sofa::core::behavior::ConstraintSolver>(core::objectmodel::BaseContext::SearchDown));
    if (!l_constraintSolver)
    {
        if (const auto constraintSolver = sofa::core::objectmodel::New<constraint::lagrangian::solver::BlockGaussSeidelConstraintSolver>())
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

void registerFreeMotionAnimationLoop(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData(R"(
The animation loop to use with constraints.
You must add this loop at the beginning of the scene if you are using constraints.")")
        .add< FreeMotionAnimationLoop >());
}

} //namespace sofa::component::animationloop
