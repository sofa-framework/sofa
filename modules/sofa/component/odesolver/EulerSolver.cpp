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
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/helper/Quater.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>


#define SOFA_NO_VMULTIOP




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
    :
    symplectic( initData( &symplectic, true, "symplectic", "If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{
}

typedef simulation::Node::ctime_t ctime_t;
void EulerSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector acc(this, VecId::dx());
    MultiVector f(this, VecId::force());

    //---------------------------------------------------------------
    //DEBUGGING TOOLS
// 	bool printLog = f_printLog.getValue();
    //---------------------------------------------------------------

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    computeForce(f);
// 	if( printLog )
// 	  {
// 	    serr<<"EulerSolver, dt = "<< dt <<sendl;
// 	    serr<<"EulerSolver, initial x = "<< pos <<sendl;
// 	    serr<<"EulerSolver, initial v = "<< vel <<sendl;
// 	    serr<<"EulerSolver, f = "<< f <<sendl;
// 	  }
    accFromF(acc, f);
    projectResponse(acc);
// 	if( printLog )
// 	  {
// 	    serr<<"EulerSolver, a = "<< acc <<sendl;
// 	  }


    solveConstraint(dt,acc,core::behavior::BaseConstraintSet::ACC);

    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if( symplectic.getValue() )
    {
        vel.peq(acc,dt);
        solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
        pos.peq(vel,dt);
        solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);

    }
    else
    {
        pos.peq(vel,dt);
        solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
        vel.peq(acc,dt);
        solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
    }
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // change order of operations depending on the symplectic flag
        int op_vel = (symplectic.getValue()?0:1);
        int op_pos = (symplectic.getValue()?1:0);
        ops[op_vel].first = (VecId)vel;
        ops[op_vel].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[op_vel].second.push_back(std::make_pair((VecId)acc,dt));
        ops[op_pos].first = (VecId)pos;
        ops[op_pos].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[op_pos].second.push_back(std::make_pair((VecId)vel,dt));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());

        solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
        solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
    }
#endif

// 	if( printLog )
// 	  {
// 	    serr<<"EulerSolver, final x = "<< pos <<sendl;
// 	    serr<<"EulerSolver, final v = "<< vel <<sendl;
// 	  }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

