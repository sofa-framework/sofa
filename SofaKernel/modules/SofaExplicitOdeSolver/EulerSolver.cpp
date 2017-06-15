/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <math.h>
#include <iostream>
#include <sofa/helper/AdvancedTimer.h>
using std::cerr;
using std::endl;

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

int EulerSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< EulerSolver >()
        .addAlias("Euler")
        .addAlias("EulerExplicit")
        .addAlias("ExplicitEuler")
        .addAlias("EulerExplicitSolver")
        .addAlias("ExplicitEulerSolver")
        ;

SOFA_DECL_CLASS(Euler);

EulerSolver::EulerSolver()
    : symplectic( initData( &symplectic, true, "symplectic", "If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{
}

typedef simulation::Visitor::ctime_t ctime_t;

void EulerSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(false); // this solver is explicit only
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv acc(&vop, core::VecDerivId::dx() ); acc.realloc( &vop, true, true ); // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );
    MultiVecCoord pos2(&vop, xResult /*core::VecCoordId::position()*/ );
    MultiVecDeriv vel2(&vop, vResult /*core::VecDerivId::velocity()*/ );

    mop.addSeparateGravity(dt); // v += dt*g . Used if mass wants to added G separately from the other forces to v.
    sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
    mop.computeForce(f);
    sofa::helper::AdvancedTimer::stepEnd("ComputeForce");

    sofa::helper::AdvancedTimer::stepBegin("AccFromF");
    mop.accFromF(acc, f);
    sofa::helper::AdvancedTimer::stepEnd("AccFromF");
    mop.projectResponse(acc);

    mop.solveConstraint(acc, core::ConstraintParams::ACC);
#ifdef SOFA_SMP
    // For SofaSMP we would need VMultiOp to be implemented in a SofaSMP compatible way
#define SOFA_NO_VMULTIOP
#endif

    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if (symplectic.getValue())
    {
        vel2.eq(vel, acc, dt);
        mop.solveConstraint(vel2, core::ConstraintParams::VEL);
        pos2.eq(pos, vel2, dt);
        mop.solveConstraint(pos2, core::ConstraintParams::POS);
    }
    else
    {
        pos2.eq(pos, vel, dt);
        mop.solveConstraint(pos2, core::ConstraintParams::POS);
        vel2.eq(vel, acc, dt);
        mop.solveConstraint(vel2, core::ConstraintParams::VEL);
    }
#else // single-operation optimization
    {
//        cerr<<"EulerSolver::solve, x = " << pos << endl;
//        cerr<<"EulerSolver::solve, v = " << vel << endl;
//        cerr<<"EulerSolver::solve, a = " << acc << endl;

        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // change order of operations depending on the symplectic flag
        int op_vel = (symplectic.getValue()?0:1);
        int op_pos = (symplectic.getValue()?1:0);
        ops[op_vel].first = vel2;
        ops[op_vel].second.push_back(std::make_pair(vel.id(),1.0));
        ops[op_vel].second.push_back(std::make_pair(acc.id(),dt));
        ops[op_pos].first = pos2;
        ops[op_pos].second.push_back(std::make_pair(pos.id(),1.0));
        ops[op_pos].second.push_back(std::make_pair(vel2.id(),dt));

        vop.v_multiop(ops);

        mop.solveConstraint(vel2,core::ConstraintParams::VEL);
        mop.solveConstraint(pos2,core::ConstraintParams::POS);

//        cerr<<"EulerSolver::solve, new x = " << pos << endl;
//        cerr<<"EulerSolver::solve, new v = " << vel << endl;
    }
#endif
}

} // namespace odesolver

} // namespace component

} // namespace sofa
