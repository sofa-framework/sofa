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
#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::odesolver::backward
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

EulerImplicitSolver::EulerImplicitSolver()
    : d_rayleighStiffness(initData(&d_rayleighStiffness, (SReal)0.0, "rayleighStiffness", "Rayleigh damping coefficient related to stiffness, > 0") )
    , d_rayleighMass(initData(&d_rayleighMass, (SReal)0.0, "rayleighMass", "Rayleigh damping coefficient related to mass, > 0"))
    , d_velocityDamping(initData(&d_velocityDamping, (SReal)0.0, "vdamping", "Velocity decay coefficient (no decay if null)") )
    , d_firstOrder (initData(&d_firstOrder, false, "firstOrder", "Use backward Euler scheme for first order ODE system, which means that only the first derivative of the DOFs (state) appears in the equation. Higher derivatives are absent"))
    , d_trapezoidalScheme( initData(&d_trapezoidalScheme,false,"trapezoidalScheme","Boolean to use the trapezoidal scheme instead of the implicit Euler scheme and get second order accuracy in time (false by default)") )
    , d_solveConstraint(initData(&d_solveConstraint, false, "solveConstraint", "Apply ConstraintSolver (requires a ConstraintSolver in the same node as this solver, disabled by by default for now)") )
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
{
    f_rayleighStiffness.setOriginalData(&d_rayleighStiffness);
    f_rayleighMass.setOriginalData(&d_rayleighMass);
    f_velocityDamping.setOriginalData(&d_velocityDamping);
    f_firstOrder.setOriginalData(&d_firstOrder);
    f_solveConstraint.setOriginalData(&d_solveConstraint);

}

void EulerImplicitSolver::init()
{
    if (!this->getTags().empty())
    {
        msg_info() << "EulerImplicitSolver: responsible for the following objects with tags " << this->getTags() << " :";
        type::vector<core::objectmodel::BaseObject*> objs;
        this->getContext()->get<core::objectmodel::BaseObject>(&objs,this->getTags(),sofa::core::objectmodel::BaseContext::SearchDown);
        for (const auto* obj : objs)
        {
            msg_info() << "  " << obj->getClassName() << ' ' << obj->getName();
        }
    }
    sofa::core::behavior::OdeSolver::init();
    sofa::core::behavior::LinearSolverAccessor::init();
}

void EulerImplicitSolver::cleanup()
{
    // free the locally created vector x (including eventual external mechanical states linked by an InteractionForceField)
    sofa::simulation::common::VectorOperations vop( core::execparams::defaultInstance(), this->getContext() );
    vop.v_free(x.id(), !d_threadSafeVisitor.getValue(), true);
}

void EulerImplicitSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SolverVectorAllocation");
#endif
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv f(&vop, core::VecDerivId::force() );
    MultiVecDeriv b(&vop, true, core::VecIdProperties{"RHS", GetClass()->className});
    MultiVecCoord newPos(&vop, xResult );
    MultiVecDeriv newVel(&vop, vResult );

    /// inform the constraint parameters about the position and velocity id
    mop.cparams.setX(xResult);
    mop.cparams.setV(vResult);

    // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)
    MultiVecDeriv dx(&vop, core::VecDerivId::dx());
    dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    x.realloc(&vop, !d_threadSafeVisitor.getValue(), true, core::VecIdProperties{"solution", GetClass()->className});


#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SolverVectorAllocation");
#endif


    const SReal& h = dt;
    const bool firstOrder = d_firstOrder.getValue();

    // the only difference for the trapezoidal rule is the factor tr = 0.5 for some usages of h
    const bool optTrapezoidal = d_trapezoidalScheme.getValue();
    SReal tr;
    if (optTrapezoidal)
        tr = 0.5;
    else
        tr = 1.0;

    msg_info() << "trapezoidal factor = " << tr;

    {
        SCOPED_TIMER("ComputeForce");
        mop->setImplicit(true); // this solver is implicit
        // compute the net forces at the beginning of the time step
        mop.computeForce(f);                                                               //f = Kx + Bv

        msg_info() << "EulerImplicitSolver, initial f = " << f;
    }

    {
        SCOPED_TIMER("ComputeRHTerm");

        if (firstOrder)
        {
            b.eq(f);
        }
        else
        {
            // new more powerful visitors

            // force in the current configuration
            b.eq(f, 1.0 / tr);                                                                    // b = f

            msg_info() << "EulerImplicitSolver, f = " << f;

            // add the change of force due to stiffness + Rayleigh damping
            mop.addMBKv(b, -d_rayleighMass.getValue(), 0,
                        h + d_rayleighStiffness.getValue()); // b =  f + ( rm M + (h+rs) K ) v

            // integration over a time step
            b.teq(h *
                  tr);                                                                       // b = h(f + ( rm M + (h+rs) K ) v )
        }

        msg_info() << "EulerImplicitSolver, b = " << b;

        mop.projectResponse(b);          // b is projected to the constrained space

        msg_info() << "EulerImplicitSolver, projected b = " << b;
    }

    {
        SCOPED_TIMER("setSystemMBKMatrix");
        SReal mFact, kFact, bFact;
        if (firstOrder)
        {
            mFact = 1;
            bFact = 0;
            kFact = -h * tr;
        }
        else
        {
            mFact = 1 + tr * h * d_rayleighMass.getValue();
            bFact = -tr * h;
            kFact = -tr * h * (h + d_rayleighStiffness.getValue());
        }
        mop.setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("SystemSolution");
#endif
    }

    {
        SCOPED_TIMER("MBKSolve");

        l_linearSolver->setSystemLHVector(x);
        l_linearSolver->setSystemRHVector(b);
        l_linearSolver->solveSystem();
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("SystemSolution");
#endif

    // mop.projectResponse(x);
    // x is the solution of the system
    // apply the solution

    const bool solveConstraint = d_solveConstraint.getValue();

#ifndef SOFA_NO_VMULTIOP // unoptimized version
    if (solveConstraint)
#endif
    {
    if (firstOrder)
    {
        const char* prevStep = "UpdateV";
        sofa::helper::AdvancedTimer::stepBegin(prevStep);
#define SOFATIMER_NEXTSTEP(s) { sofa::helper::AdvancedTimer::stepNext(prevStep,s); prevStep=s; }

        newVel.eq(x);                       // vel = x

        if (solveConstraint)
        {
            SOFATIMER_NEXTSTEP("CorrectV");
            mop.solveConstraint(newVel,core::ConstraintOrder::VEL);
        }
        SOFATIMER_NEXTSTEP("UpdateX");

        newPos.eq(pos, newVel, h);          // pos = pos + h vel

        if (solveConstraint)
        {
            SOFATIMER_NEXTSTEP("CorrectX");
            mop.solveConstraint(newPos,core::ConstraintOrder::POS);
        }
#undef SOFATIMER_NEXTSTEP
        sofa::helper::AdvancedTimer::stepEnd  (prevStep);
    }
    else
    {
        const char* prevStep = "UpdateV";
        sofa::helper::AdvancedTimer::stepBegin(prevStep);
#define SOFATIMER_NEXTSTEP(s) { sofa::helper::AdvancedTimer::stepNext(prevStep,s); prevStep=s; }

        // vel = vel + x
        newVel.eq(vel, x);

        if (solveConstraint)
        {
            SOFATIMER_NEXTSTEP("CorrectV");
            mop.solveConstraint(newVel,core::ConstraintOrder::VEL);
        }
        SOFATIMER_NEXTSTEP("UpdateX");

        // pos = pos + h vel
        newPos.eq(pos, newVel, h);

        if (solveConstraint)
        {
            SOFATIMER_NEXTSTEP("CorrectX");
            mop.solveConstraint(newPos,core::ConstraintOrder::POS);
        }
#undef SOFATIMER_NEXTSTEP
        sofa::helper::AdvancedTimer::stepEnd  (prevStep);
    }

    }
#ifndef SOFA_NO_VMULTIOP
    else
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        if (firstOrder)
        {
            ops.resize(2);
            ops[0].first = newVel;
            ops[0].second.push_back(std::make_pair(x.id(),1.0));
            ops[1].first = newPos;
            ops[1].second.push_back(std::make_pair(pos.id(),1.0));
            ops[1].second.push_back(std::make_pair(newVel.id(),h));
        }
        else
        {
            ops.resize(2);
            ops[0].first = newVel;
            ops[0].second.push_back(std::make_pair(vel.id(),1.0));
            ops[0].second.push_back(std::make_pair(x.id(),1.0));
            ops[1].first = newPos;
            ops[1].second.push_back(std::make_pair(pos.id(),1.0));
            ops[1].second.push_back(std::make_pair(newVel.id(),h));
        }

        SCOPED_TIMER_VARNAME(updateVAndXTimer, "UpdateVAndX");
        vop.v_multiop(ops);
        if (solveConstraint)
        {
            {
                SCOPED_TIMER_VARNAME(correctVTimer, "CorrectV");
                mop.solveConstraint(newVel,core::ConstraintOrder::VEL);
            }
            {
                SCOPED_TIMER_VARNAME(correctXTimer, "CorrectX");
                mop.solveConstraint(newPos,core::ConstraintOrder::POS);
            }
        }
    }
#endif

    mop.addSeparateGravity(dt, newVel);	// v += dt*g . Used if mass wants to add G separately from the other forces to v

    if (d_velocityDamping.getValue() != 0.0)
        newVel *= exp(-h * d_velocityDamping.getValue());

    if( f_printLog.getValue() )
    {
        mop.projectPosition(newPos);
        mop.projectVelocity(newVel);
        mop.propagateX(newPos);
        mop.propagateV(newVel);
        msg_info() << "EulerImplicitSolver, final x = " << newPos;
        msg_info() << "EulerImplicitSolver, final v = " << newVel;
        mop.computeForce(f);
        msg_info() << "EulerImplicitSolver, final f = " << f;
    }
}


SReal EulerImplicitSolver::getPositionIntegrationFactor() const
{
    return getPositionIntegrationFactor(getContext()->getDt());
}

SReal EulerImplicitSolver::getIntegrationFactor(int inputDerivative, int outputDerivative) const
{
    return getIntegrationFactor(inputDerivative, outputDerivative, getContext()->getDt());
}

SReal EulerImplicitSolver::getIntegrationFactor(int inputDerivative, int outputDerivative, SReal dt) const
{
    const SReal matrix[3][3] =
    {
        { 1, dt, 0},
        { 0, 1, 0},
        { 0, 0, 0}
    };
    if (inputDerivative >= 3 || outputDerivative >= 3)
        return 0;
    else
        return matrix[outputDerivative][inputDerivative];
}

SReal EulerImplicitSolver::getSolutionIntegrationFactor(int outputDerivative) const
{
    return getSolutionIntegrationFactor(outputDerivative, getContext()->getDt());
}

SReal EulerImplicitSolver::getSolutionIntegrationFactor(int outputDerivative, SReal dt) const
{
    const SReal vect[3] = { dt, 1, 1/dt};
    if (outputDerivative >= 3)
        return 0;
    else
        return vect[outputDerivative];
}


int EulerImplicitSolverClass = core::RegisterObject("Time integrator using implicit backward Euler scheme")
        .add< EulerImplicitSolver >()
        .addAlias("EulerImplicit")
        .addAlias("ImplicitEulerSolver")
        .addAlias("ImplicitEuler")
        ;

} // namespace sofa::component::odesolver::backward
