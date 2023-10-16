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
#include <sofa/core/behavior/MultiMatrix.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::odesolver::backward
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

EulerImplicitSolver::EulerImplicitSolver()
    : f_rayleighStiffness( initData(&f_rayleighStiffness,(SReal)0.0,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, > 0") )
    , f_rayleighMass( initData(&f_rayleighMass,(SReal)0.0,"rayleighMass","Rayleigh damping coefficient related to mass, > 0"))
    , f_velocityDamping( initData(&f_velocityDamping,(SReal)0.0,"vdamping","Velocity decay coefficient (no decay if null)") )
    , f_firstOrder (initData(&f_firstOrder, false, "firstOrder", "Use backward Euler scheme for first order ode system."))
    , d_trapezoidalScheme( initData(&d_trapezoidalScheme,false,"trapezoidalScheme","Optional: use the trapezoidal scheme instead of the implicit Euler scheme and get second order accuracy in time") )
    , f_solveConstraint( initData(&f_solveConstraint,false,"solveConstraint","Apply ConstraintSolver (requires a ConstraintSolver in the same node as this solver, disabled by by default for now)") )
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
{
}

void EulerImplicitSolver::init()
{
    if (!this->getTags().empty())
    {
        msg_info() << "EulerImplicitSolver: responsible for the following objects with tags " << this->getTags() << " :";
        type::vector<core::objectmodel::BaseObject*> objs;
        this->getContext()->get<core::objectmodel::BaseObject>(&objs,this->getTags(),sofa::core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<objs.size(); ++i)
            msg_info() << "  " << objs[i]->getClassName() << ' ' << objs[i]->getName();
    }
    sofa::core::behavior::OdeSolver::init();
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
    const bool firstOrder = f_firstOrder.getValue();

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
        mop.computeForce(f);

        msg_info() << "EulerImplicitSolver, initial f = " << f;
    }

    sofa::helper::AdvancedTimer::stepBegin("ComputeRHTerm");
    if( firstOrder )
    {
        b.eq(f);
    }
    else
    {
        // new more powerful visitors

        // force in the current configuration
        b.eq(f,1.0/tr);                                                                         // b = f0

        msg_info() << "EulerImplicitSolver, f = " << f;

        // add the change of force due to stiffness + Rayleigh damping
        mop.addMBKv(b, -f_rayleighMass.getValue(), 1, h+f_rayleighStiffness.getValue()); // b =  f0 + ( rm M + B + (h+rs) K ) v

        // integration over a time step
        b.teq(h*tr);                                                                        // b = h(f0 + ( rm M + B + (h+rs) K ) v )
    }

    msg_info() << "EulerImplicitSolver, b = " << b;

    mop.projectResponse(b);          // b is projected to the constrained space

    msg_info() << "EulerImplicitSolver, projected b = " << b;

    sofa::helper::AdvancedTimer::stepNext ("ComputeRHTerm", "MBKBuild");

    core::behavior::MultiMatrix<simulation::common::MechanicalOperations> matrix(&mop);

    if (firstOrder)
        matrix.setSystemMBKMatrix(MechanicalMatrix(1,0,-h*tr)); //MechanicalMatrix::K * (-h*tr) + MechanicalMatrix::M;
    else
        matrix.setSystemMBKMatrix(MechanicalMatrix(1+tr*h*f_rayleighMass.getValue(),-tr*h,-tr*h*(h+f_rayleighStiffness.getValue()))); // MechanicalMatrix::K * (-tr*h*(h+f_rayleighStiffness.getValue())) + MechanicalMatrix::B * (-tr*h) + MechanicalMatrix::M * (1+tr*h*f_rayleighMass.getValue());

    msg_info() << "EulerImplicitSolver, matrix = " << (MechanicalMatrix::K * (-h * (h + f_rayleighStiffness.getValue())) + MechanicalMatrix::M * (1 + h * f_rayleighMass.getValue())) << " = " << matrix;
    msg_info() << "EulerImplicitSolver, Matrix K = " << MechanicalMatrix::K;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("SystemSolution");
#endif
    sofa::helper::AdvancedTimer::stepEnd ("MBKBuild");
    {
        SCOPED_TIMER("MBKSolve");
        matrix.solve(x, b); //Call to ODE resolution: x is the solution of the system}
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("SystemSolution");
#endif

    // mop.projectResponse(x);
    // x is the solution of the system
    // apply the solution

    const bool solveConstraint = f_solveConstraint.getValue();

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

    if (f_velocityDamping.getValue()!=0.0)
        newVel *= exp(-h*f_velocityDamping.getValue());

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
