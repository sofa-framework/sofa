/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/MechanicalVisitor.h>
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

int EulerSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< EulerSolver >()
        .addAlias("Euler")
        ;

SOFA_DECL_CLASS(Euler);

EulerSolver::EulerSolver()
    : symplectic( initData( &symplectic, true, "symplectic", "If true, the velocities are updated before the velocities and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{}

void EulerSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector acc(this, VecId::dx());
    MultiVector f(this, VecId::force());
    bool printLog = f_printLog.getValue();


    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    computeForce(f);
    if( printLog )
    {
        cerr<<"EulerSolver, dt = "<< dt <<endl;
        cerr<<"EulerSolver, initial x = "<< pos <<endl;
        cerr<<"EulerSolver, initial v = "<< vel <<endl;
        cerr<<"EulerSolver, f = "<< f <<endl;
    }
    accFromF(acc, f);
    projectResponse(acc);
    if( printLog )
    {
        cerr<<"EulerSolver, a = "<< acc <<endl;
    }

    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if( symplectic.getValue() )
    {
        vel.peq(acc,dt);
        pos.peq(vel,dt);
    }
    else
    {
        pos.peq(vel,dt);
        vel.peq(acc,dt);
    }
#else // single-operation optimization
    {
        simulation::MechanicalVMultiOpVisitor vmop;
        vmop.ops.resize(2);
        // change order of operations depending on the sympletic flag
        int op_vel = (symplectic.getValue()?0:1);
        int op_pos = (symplectic.getValue()?1:0);
        vmop.ops[op_vel].first = (VecId)vel;
        vmop.ops[op_vel].second.push_back(std::make_pair((VecId)vel,1.0));
        vmop.ops[op_vel].second.push_back(std::make_pair((VecId)acc,dt));
        vmop.ops[op_pos].first = (VecId)pos;
        vmop.ops[op_pos].second.push_back(std::make_pair((VecId)pos,1.0));
        vmop.ops[op_pos].second.push_back(std::make_pair((VecId)vel,dt));
        vmop.execute(this->getContext());
    }
#endif

    if( printLog )
    {
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

