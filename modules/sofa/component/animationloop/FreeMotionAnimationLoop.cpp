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
#include <sofa/component/animationloop/FreeMotionAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/component/constraintset/LCPConstraintSolver.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/SolveVisitor.h>
#include <sofa/simulation/common/VectorOperations.h>


namespace sofa
{

namespace component
{

namespace animationloop
{

using namespace core::behavior;

FreeMotionAnimationLoop::FreeMotionAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , m_solveVelocityConstraintFirst(initData(&m_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
    , constraintSolver(NULL)
    , defaultSolver(NULL)
{
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
{
    if (defaultSolver != NULL)
        delete defaultSolver;
}

void FreeMotionAnimationLoop::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    this->simulation::CollisionAnimationLoop::parse(arg);

    defaultSolver = new constraintset::LCPConstraintSolver;
    defaultSolver->parse(arg);
}


void FreeMotionAnimationLoop::init()
{
    getContext()->get(constraintSolver, core::objectmodel::BaseContext::SearchDown);
    if (constraintSolver == NULL && defaultSolver != NULL)
    {
        serr << "No ConstraintSolver found, using default LCPConstraintSolver" << sendl;
        this->getContext()->addObject(defaultSolver);
        constraintSolver = defaultSolver;
        defaultSolver = NULL;
    }
    else
    {
        delete defaultSolver;
        defaultSolver = NULL;
    }
}


void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params /* PARAMS FIRST */, double dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();

    sofa::helper::AdvancedTimer::stepBegin("AnimationStep");
    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    double startTime = this->gnode->getTime();
    double mechanicalDt = dt / numMechSteps.getValue();

    simulation::common::VectorOperations vop(params, this->getContext());
    simulation::common::MechanicalOperations mop(params, this->getContext());

    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecCoord freePos(&vop, core::VecCoordId::freePosition() );
    MultiVecDeriv freeVel(&vop, core::VecDerivId::freeVelocity() );

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    simulation::MechanicalVInitVisitor< core::V_COORD >(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(this->gnode);
    simulation::MechanicalVInitVisitor< core::V_DERIV >(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity(), true).execute(this->gnode);

    BehaviorUpdatePositionVisitor beh(params , this->gnode->getDt());

    for (unsigned i = 0; i < numMechSteps.getValue(); i++ )
    {
        using helper::system::thread::CTime;
        using sofa::helper::AdvancedTimer;

        double time = 0.0;
        double timeTotal = 0.0;
        double timeScale = 1000.0 / (double)CTime::getTicksPerSec();

        if (displayTime.getValue())
        {
            time = (double) CTime::getTime();
            timeTotal = (double) CTime::getTime();
        }

        // Update the BehaviorModels
        // Required to allow the RayPickInteractor interaction
        if (f_printLog.getValue())
            serr << "updatePos called" << sendl;

        AdvancedTimer::stepBegin("UpdatePosition");
        this->gnode->execute(&beh);
        AdvancedTimer::stepEnd("UpdatePosition");

        if (f_printLog.getValue())
            serr << "updatePos performed - beginVisitor called" << sendl;

        simulation::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
        this->gnode->execute(&beginVisitor);

        if (f_printLog.getValue())
            serr << "beginVisitor performed - SolveVisitor for freeMotion is called" << sendl;

        // Free Motion
        AdvancedTimer::stepBegin("FreeMotion");
        simulation::SolveVisitor freeMotion(params, dt, true);
        this->gnode->execute(&freeMotion);
        AdvancedTimer::stepEnd("FreeMotion");

        mop.propagateXAndV(freePos, freeVel);

        if (f_printLog.getValue())
            serr << " SolveVisitor for freeMotion performed" << sendl;

        if (displayTime.getValue())
        {
            sout << " >>>>> Begin display FreeMotionAnimationLoop time" << sendl;
            sout <<" Free Motion " << ((double)CTime::getTime() - time) * timeScale << " ms" << sendl;

            time = (double)CTime::getTime();
        }

        // Collision detection and response creation
        AdvancedTimer::stepBegin("Collision");
        computeCollision(params);
        AdvancedTimer::stepEnd  ("Collision");

        mop.propagateX(pos);

        if (displayTime.getValue())
        {
            sout << " computeCollision " << ((double) CTime::getTime() - time) * timeScale << " ms" << sendl;
            time = (double)CTime::getTime();
        }

        // Solve constraints
        if (constraintSolver)
        {
            AdvancedTimer::stepBegin("ConstraintSolver");

            if (m_solveVelocityConstraintFirst.getValue())
            {
                core::ConstraintParams cparams(*params);
                cparams.setX(freePos);
                cparams.setV(freeVel);

                cparams.setOrder(core::ConstraintParams::VEL);
                constraintSolver->solveConstraint(&cparams, vel);

                MultiVecDeriv dv(&vop, constraintSolver->getDx());
                mop.propagateDx(dv);

                // xfree += dv * dt
                freePos.eq(freePos, dv, this->getContext()->getDt());
                mop.propagateX(freePos);

                cparams.setOrder(core::ConstraintParams::POS);
                constraintSolver->solveConstraint(&cparams, pos);

                MultiVecDeriv dx(&vop, constraintSolver->getDx());

                mop.propagateV(vel);
                mop.propagateDx(dx);

                // "mapped" x = xfree + dx
                simulation::MechanicalVOpVisitor(params, pos, freePos, dx, 1.0 ).setOnlyMapped(true).execute(this->gnode);
            }
            else
            {
                core::ConstraintParams cparams(*params);
                cparams.setX(freePos);
                cparams.setV(freeVel);

                constraintSolver->solveConstraint(&cparams, pos, vel);

                mop.propagateV(vel);

                MultiVecDeriv dx(&vop, constraintSolver->getDx());
                mop.propagateDx(dx);

                // "mapped" x = xfree + dx
                simulation::MechanicalVOpVisitor(params, pos, freePos, dx, 1.0 ).setOnlyMapped(true).execute(this->gnode);
            }
        }

        if ( displayTime.getValue() )
        {
            sout << " contactCorrections " << ((double)CTime::getTime() - time) * timeScale << " ms" <<sendl;
            sout << "<<<<<< End display FreeMotionAnimationLoop time." << sendl;
        }

        simulation::MechanicalEndIntegrationVisitor endVisitor(params /* PARAMS FIRST */, dt);
        this->gnode->execute(&endVisitor);

        this->gnode->setTime ( startTime + (i+1) * dt );
        this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        nbMechSteps.setValue(nbMechSteps.getValue() + 1);
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
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);

    sofa::helper::AdvancedTimer::stepEnd("AnimationStep");
}


SOFA_DECL_CLASS(FreeMotionAnimationLoop)

int FreeMotionAnimationLoopClass = core::RegisterObject("Constraint solver")
        .add< FreeMotionAnimationLoop >()
        ;

} // namespace animationloop

} // namespace component

} // namespace sofa
