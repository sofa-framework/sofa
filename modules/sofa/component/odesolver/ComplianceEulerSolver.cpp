/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/odesolver/ComplianceEulerSolver.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

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
    OdeSolver* group = this;
    MultiVector force(group, VecId::force());
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector acc(group, VecId::dx());
    MultiVector posFree(group, VecId::freePosition());
    MultiVector velFree(group, VecId::freeVelocity());
    MultiVector dx(group, VecId::V_DERIV);

    bool printLog = f_printLog.getValue();

    simulation::tree::GNode *context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

    if( printLog )
    {
        cerr<<"ComplianceEulerSolver, dt = "<< dt <<endl;
        cerr<<"ComplianceEulerSolver, initial x = "<< pos <<endl;
        cerr<<"ComplianceEulerSolver, initial v = "<< vel <<endl;
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

    computeAcc(getTime(), acc, pos, vel);
    vel.eq(vel);
    vel.peq(acc,dt);
    pos.eq(pos);
    pos.peq(vel,dt);
    //simulation::tree::MechanicalPropagateFreePositionVisitor().execute(context);

//}


    firstCallToSolve.setValue(!firstCallToSolve.getValue());

    if( printLog )
    {
        cerr<<"ComplianceEulerSolver, acceleration = "<< acc <<endl;
        cerr<<"ComplianceEulerSolver, final x = "<< pos <<endl;
        cerr<<"ComplianceEulerSolver, final v = "<< vel <<endl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

