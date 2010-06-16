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
#include <sofa/component/odesolver/RungeKutta2Solver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>




namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace core::behavior;
using namespace sofa::defaulttype;

int RungeKutta2SolverClass = core::RegisterObject("A popular explicit time integrator")
        .add< RungeKutta2Solver >()
        .addAlias("RungeKutta2")
        ;

SOFA_DECL_CLASS(RungeKutta2);


void RungeKutta2Solver::solve(double dt)
{
    // Get the Ids of the state vectors
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());

    // Allocate auxiliary vectors
    MultiVector acc(this, VecId::V_DERIV);
    MultiVector newX(this, VecId::V_COORD);
    MultiVector newV(this, VecId::V_DERIV);

    double startTime = this->getTime();

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    // Compute state derivative. vel is the derivative of pos
    computeAcc (startTime, acc, pos, vel); // acc is the derivative of vel

    // Perform a dt/2 step along the derivative
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    newX = pos;
    newX.peq(vel, dt/2.); // newX = pos + vel dt/2
    newV = vel;
    newV.peq(acc, dt/2.); // newV = vel + acc dt/2
#else // single-operation optimization
    {

        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)vel,dt/2));
        ops[1].first = (VecId)newV;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)acc,dt/2));

        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    // Compute the derivative at newX, newV
    computeAcc ( startTime+dt/2., acc, newX, newV);

    // Use the derivative at newX, newV to update the state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    pos.peq(newV,dt);
    solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
    vel.peq(acc,dt);
    solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = (VecId)pos;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)newV,dt));
        ops[1].first = (VecId)vel;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)acc,dt));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());

        solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
        solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
    }
#endif


}



} // namespace odesolver

} // namespace component

} // namespace sofa

