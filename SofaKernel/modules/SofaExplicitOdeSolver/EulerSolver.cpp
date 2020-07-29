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
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/helper/Quater.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MultiVec.h>
#include <cmath>
#include <iostream>
#include <sofa/helper/AdvancedTimer.h>

//#define SOFA_NO_VMULTIOP

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

int EulerExplicitSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< EulerExplicitSolver >()
        .addAlias("Euler")
        .addAlias("EulerExplicit")
        .addAlias("ExplicitEuler")
        .addAlias("EulerSolver")
        .addAlias("ExplicitEulerSolver")
        ;

EulerExplicitSolver::EulerExplicitSolver()
    : d_symplectic( initData( &d_symplectic, true, "symplectic", "If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
    , d_optimizedForDiagonalMatrix(initData(&d_optimizedForDiagonalMatrix, true, "optimizedForDiagonalMatrix", "If true, solution to the system Ax=b can be directly found by computing x = f/m. Must be set to false if M is sparse."))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
{
}

typedef simulation::Visitor::ctime_t ctime_t;

void EulerExplicitSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(false); // this solver is explicit only
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );

    MultiVecCoord newPos(&vop, xResult /*core::VecCoordId::position()*/ );
    MultiVecDeriv newVel(&vop, vResult /*core::VecDerivId::velocity()*/ );
    MultiVecDeriv acc(&vop, core::VecDerivId::dx());

    acc.realloc(&vop, !d_threadSafeVisitor.getValue(), true); // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)

    // Mass matrix is diagonal, solution can thus be found by computing acc = f/m
    if(d_optimizedForDiagonalMatrix.getValue())
    {
        mop.addSeparateGravity(dt); // v += dt*g . Used if mass wants to add G separately from the other forces to v.
        sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
        mop.computeForce(f);
        sofa::helper::AdvancedTimer::stepEnd("ComputeForce");

        sofa::helper::AdvancedTimer::stepBegin("AccFromF");
        mop.accFromF(acc, f);
        sofa::helper::AdvancedTimer::stepEnd("AccFromF");
        mop.projectResponse(acc);

        mop.solveConstraint(acc, core::ConstraintParams::ACC);


    }
    else
    {
        x.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

        mop.addSeparateGravity(dt); // v += dt*g . Used if mass wants to added G separately from the other forces to v.

        sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
        mop.computeForce(f);
        sofa::helper::AdvancedTimer::stepEnd("ComputeForce");

        sofa::helper::AdvancedTimer::stepBegin ("projectResponse");
        mop.projectResponse(f);
        sofa::helper::AdvancedTimer::stepEnd ("projectResponse");

        sofa::helper::AdvancedTimer::stepBegin ("MBKBuild");
        core::behavior::MultiMatrix<simulation::common::MechanicalOperations> matrix(&mop);
        matrix = MechanicalMatrix(1.0,0,0); // MechanicalMatrix::M;
        sofa::helper::AdvancedTimer::stepEnd ("MBKBuild");

        sofa::helper::AdvancedTimer::stepBegin ("MBKSolve");
        matrix.solve(x, f); //Call to ODE resolution: x is the solution of the system
        sofa::helper::AdvancedTimer::stepEnd  ("MBKSolve");

        acc.eq(x);
    }


    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if (d_symplectic.getValue())
    {
        newVel.eq(vel, acc, dt);
        mop.solveConstraint(newVel, core::ConstraintParams::VEL);
        newPos.eq(pos, newVel, dt);
        mop.solveConstraint(newPos, core::ConstraintParams::POS);
    }
    else
    {
        newPos.eq(pos, vel, dt);
        mop.solveConstraint(newPos, core::ConstraintParams::POS);
        newVel.eq(vel, acc, dt);
        mop.solveConstraint(newVel, core::ConstraintParams::VEL);
    }
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // change order of operations depending on the symplectic flag
        size_t op_vel = (d_symplectic.getValue()?0:1);
        size_t op_pos = (d_symplectic.getValue()?1:0);
        ops[op_vel].first = newVel;
        ops[op_vel].second.push_back(std::make_pair(vel.id(),1.0));
        ops[op_vel].second.push_back(std::make_pair(acc.id(),dt));
        ops[op_pos].first = newPos;
        ops[op_pos].second.push_back(std::make_pair(pos.id(),1.0));
        ops[op_pos].second.push_back(std::make_pair(newVel.id(),dt));

        vop.v_multiop(ops);

        mop.solveConstraint(newVel,core::ConstraintParams::VEL);
        mop.solveConstraint(newPos,core::ConstraintParams::POS);
    }
#endif
}

double EulerExplicitSolver::getIntegrationFactor(int inputDerivative, int outputDerivative) const
{
    const SReal dt = getContext()->getDt();
    double matrix[3][3] =
    {
        { 1, dt, ((d_symplectic.getValue())?dt*dt:0.0)},
        { 0, 1, dt},
        { 0, 0, 0}
    };
    if (inputDerivative >= 3 || outputDerivative >= 3)
        return 0;
    else
        return matrix[outputDerivative][inputDerivative];
}

double EulerExplicitSolver::getSolutionIntegrationFactor(int outputDerivative) const
{
    const SReal dt = getContext()->getDt();
    double vect[3] = {((d_symplectic.getValue()) ? dt * dt : 0.0), dt, 1};
    if (outputDerivative >= 3)
        return 0;
    else
        return vect[outputDerivative];
}

void EulerExplicitSolver::init()
{
    OdeSolver::init();
    reinit();
}


} // namespace odesolver

} // namespace component

} // namespace sofa
