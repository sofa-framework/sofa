/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/component/odesolver/ComplianceEulerSolver.h>
#include <sofa/core/visual/VisualParams.h>
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

using namespace sofa::defaulttype;
using namespace core::behavior;

int ComplianceEulerSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< ComplianceEulerSolver >()
        .addAlias("ComplianceEuler")
        ;

SOFA_DECL_CLASS(ComplianceEuler);

ComplianceEulerSolver::ComplianceEulerSolver()
    : firstCallToSolve( initData( &firstCallToSolve, true, "firstCallToSolve", "If true, the free movement is computed, if false, the constraint movement is computed"))
{
}

void ComplianceEulerSolver::solve(double dt)
{
    MultiVector force(this, core::VecDerivId::force());
    MultiVector pos(this, core::VecCoordId::position());
    MultiVector vel(this, core::VecDerivId::velocity());
    MultiVector acc(this, core::VecDerivId::dx());
    MultiVector posFree(this, core::VecCoordId::freePosition());
    MultiVector velFree(this, core::VecDerivId::freeVelocity());
    MultiVector dx(this, core::V_DERIV);

    bool printLog = f_printLog.getValue();

    if( printLog )
    {
        serr<<"ComplianceEulerSolver, dt = "<< dt <<sendl;
        serr<<"ComplianceEulerSolver, initial x = "<< pos <<sendl;
        serr<<"ComplianceEulerSolver, initial v = "<< vel <<sendl;
    }


    //if (!firstCallToSolve.getValue()) // f = contact force
    //{
    //	/*
    //	computeContactAcc(getTime(), acc, pos, vel);
    //	vel.eq(velFree); // computes velocity after a constraint movement
    //	vel.peq(acc,dt);
    //	pos.peq(vel,dt); // Computes position after a constraint movement
    //	dx.peq(acc,(dt*dt));
    //	*/

    //	simulation::tree::MechanicalPropagateAndAddDxVisitor().execute(context);
    //}
    //else // f = mass * gravity
    //{

    //computeAcc(getTime(), acc, pos, vel);
    //velFree.eq(vel);
    //velFree.peq(acc,dt);
    //posFree.eq(pos);
    //posFree.peq(velFree,dt);
    //simulation::tree::MechanicalPropagateFreePositionVisitor().execute(context);

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    computeAcc(getTime(), acc, pos, vel);
    vel.eq(vel);
    vel.peq(acc,dt);
    solveConstraint(dt,vel, core::behavior::BaseConstraintSet::VEL);
    pos.eq(pos);
    pos.peq(vel,dt);
    solveConstraint(dt,pos, core::behavior::BaseConstraintSet::POS);

    //simulation::tree::MechanicalPropagateFreePositionVisitor().execute(context);

//}


    firstCallToSolve.setValue(!firstCallToSolve.getValue());

    if( printLog )
    {
        serr<<"ComplianceEulerSolver, acceleration = "<< acc <<sendl;
        serr<<"ComplianceEulerSolver, final x = "<< pos <<sendl;
        serr<<"ComplianceEulerSolver, final v = "<< vel <<sendl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

