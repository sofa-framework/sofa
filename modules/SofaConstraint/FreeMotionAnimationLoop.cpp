/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaConstraint/FreeMotionAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaConstraint/LCPConstraintSolver.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <SofaConstraint/LCPConstraintSolver.h>


namespace sofa
{

namespace component
{

namespace animationloop
{

using namespace core::behavior;
using namespace sofa::simulation;
using helper::system::thread::CTime;
using sofa::helper::ScopedAdvancedTimer;

FreeMotionAnimationLoop::FreeMotionAnimationLoop(simulation::Node* gnode)
    : Inherit1(gnode)
    , m_solveVelocityConstraintFirst(initData(&m_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
    , constraintSolver(nullptr)
    , defaultSolver(nullptr)
{
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
{
    if (defaultSolver != nullptr)
        defaultSolver.reset();
}

void FreeMotionAnimationLoop::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    simulation::CollisionAnimationLoop::parse(arg);

    defaultSolver = sofa::core::objectmodel::New<constraintset::LCPConstraintSolver>();
    defaultSolver->parse(arg);
}


void FreeMotionAnimationLoop::init()
{
    simulation::common::VectorOperations vop(core::ExecParams::defaultInstance(), getContext());
    MultiVecDeriv dx(&vop, core::VecDerivId::dx()); dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);
    MultiVecDeriv df(&vop, core::VecDerivId::dforce()); df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    getContext()->get(constraintSolver, core::objectmodel::BaseContext::SearchDown);
    if (constraintSolver == nullptr && defaultSolver != nullptr)
    {
        msg_error() << "No ConstraintSolver found, using default LCPConstraintSolver";
        getContext()->addObject(defaultSolver);
        constraintSolver = defaultSolver.get();
        defaultSolver = nullptr;
    }
    else
    {
        defaultSolver.reset();
    }
}


void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
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
    cparams.setDx(constraintSolver->getDx());
    cparams.setLambda(constraintSolver->getLambda());
    cparams.setOrder(m_solveVelocityConstraintFirst.getValue() ? core::ConstraintParams::VEL : core::ConstraintParams::POS_AND_VEL);

    MultiVecDeriv dx(&vop, core::VecDerivId::dx()); dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);
    MultiVecDeriv df(&vop, core::VecDerivId::dforce()); df.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    {
        ScopedAdvancedTimer timer("MechanicalVInitVisitor");
        simulation::MechanicalVInitVisitor< core::V_COORD >(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(gnode);
        simulation::MechanicalVInitVisitor< core::V_DERIV >(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity(), true).execute(gnode);
    }


#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        ScopedAdvancedTimer("AnimateBeginEvent");
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }

    double time = 0.0;
    double timeScale = 1000.0 / (double)CTime::getTicksPerSec();

    if (displayTime.getValue())
    {
        time = (double) CTime::getTime();
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    dmsg_info() << "updatePos called" ;

    {
        ScopedAdvancedTimer timer("UpdatePosition");
        BehaviorUpdatePositionVisitor beh(params, dt);        
        gnode->execute(&beh);
    }

    dmsg_info() << "updatePos performed - beginVisitor called" ;

    simulation::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    gnode->execute(&beginVisitor);

    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    // Mapping geometric stiffness coming from previous lambda.
    {
        ScopedAdvancedTimer timer("lambdaMultInvDt");
        simulation::MechanicalVOpVisitor lambdaMultInvDt(params, cparams.lambda(), sofa::core::ConstMultiVecId::null(), cparams.lambda(), 1.0 / dt);
        lambdaMultInvDt.setMapped(true);
        getContext()->executeVisitor(&lambdaMultInvDt);
    }

    {
        ScopedAdvancedTimer timer("MechanicalComputeGeometricStiffness");
        simulation::MechanicalComputeGeometricStiffness geometricStiffnessVisitor(&mop.mparams, cparams.lambda());
        getContext()->executeVisitor(&geometricStiffnessVisitor);
    }

    // Free Motion
    {
        ScopedAdvancedTimer timer("FreeMotion");
        simulation::SolveVisitor freeMotion(params, dt, true);
        gnode->execute(&freeMotion);
    }

    
    mop.projectResponse(freeVel);
    mop.propagateDx(freeVel, true);

    if (cparams.constOrder() == core::ConstraintParams::POS ||
        cparams.constOrder() == core::ConstraintParams::POS_AND_VEL)
    {
        ScopedAdvancedTimer timer("freePosEqPosPlusFreeVelDt");
        simulation::MechanicalVOpVisitor freePosEqPosPlusFreeVelDt(params, freePos, pos, freeVel, dt);
        freePosEqPosPlusFreeVelDt.setMapped(true);
        getContext()->executeVisitor(&freePosEqPosPlusFreeVelDt);
    }
    dmsg_info() << " SolveVisitor for freeMotion performed" ;

    if (displayTime.getValue())
    {
        msg_info() << " >>>>> Begin display FreeMotionAnimationLoop time  " << msgendl
                   <<" Free Motion " << ((double)CTime::getTime() - time) * timeScale << " ms" ;

        time = (double)CTime::getTime();
    }

    // Collision detection and response creation
    {
        ScopedAdvancedTimer timer("Collision");
        computeCollision(params);
    }

    if (displayTime.getValue())
    {
        msg_info() << " computeCollision " << ((double) CTime::getTime() - time) * timeScale << " ms";
        time = (double)CTime::getTime();
    }

    // Solve constraints
    if (constraintSolver)
    {
        ScopedAdvancedTimer timer("ConstraintSolver");

        if (cparams.constOrder() == core::ConstraintParams::VEL )
        {
            constraintSolver->solveConstraint(&cparams, vel);
            pos.eq(pos, vel, dt);
        }
        else
        {
            constraintSolver->solveConstraint(&cparams, pos, vel);
        }

        MultiVecDeriv dx(&vop, constraintSolver->getDx());
        mop.projectResponse(dx);
        mop.propagateDx(dx, true);
    }

    if ( displayTime.getValue() )
    {
        msg_info() << " contactCorrections " << ((double)CTime::getTime() - time) * timeScale << " ms"
                << "<<<<<< End display FreeMotionAnimationLoop time.";
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(params, dt);
    gnode->execute(&endVisitor);

    mop.projectPositionAndVelocity(pos, vel);
    mop.propagateXAndV(pos, vel);

    gnode->setTime ( startTime + dt );
    gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
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

    if (!SOFA_NO_UPDATE_BBOX)
    {
        ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute<UpdateBoundingBoxVisitor>(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif

}

static int FreeMotionAnimationLoopClass = core::RegisterObject(R"(
The animation loop to use with constraints.
You must add this loop at the beginning of the scene if you are using constraints.")")
        .add< FreeMotionAnimationLoop >()
        .addAlias("FreeMotionMasterSolver");

} // namespace animationloop

} // namespace component

} // namespace sofa
