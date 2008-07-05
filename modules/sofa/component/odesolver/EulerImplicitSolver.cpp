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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"

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

EulerImplicitSolver::EulerImplicitSolver()
    : f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
    , f_velocityDamping( initData(&f_velocityDamping,0.,"vdamping","Velocity decay coefficient (no decay if null)") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
{
}

void EulerImplicitSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector f(this, VecId::force());
    MultiVector b(this, VecId::V_DERIV);
    //MultiVector p(this, VecId::V_DERIV);
    //MultiVector q(this, VecId::V_DERIV);
    //MultiVector q2(this, VecId::V_DERIV);
    //MultiVector r(this, VecId::V_DERIV);
    MultiVector x(this, VecId::V_DERIV);

    double h = dt;
    //const bool printLog = f_printLog.getValue();
    const bool verbose  = f_verbose.getValue();

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    //projectResponse(vel);          // initial velocities are projected to the constrained space

#if 1

    // compute the right-hand term of the equation system
    // accumulation through mappings is disabled as it will be done by addMBKv after all factors are computed
    computeForce(b, true, false);             // b = f0

    //computeDfV(f);                // f = df/dx v
    //b.peq(f,h+f_rayleighStiffness.getValue());      // b = f0 + (h+rs)df/dx v
    //addMdx(b,vel,-f_rayleighMass.getValue()); // no need to propagate vel as dx again
    //f.teq(-1);
    //addMBKv(f, 0 /* (f_rayleighMass.getValue() == 0.0 ? 0.0 : -f_rayleighMass.getValue()) */, 0, 1);
    //cerr<<"EulerImplicitSolver, diff = "<< f <<endl;

    // new more powerful visitors
    // b += (h+rs)df/dx v - rd M v
    // values are not cleared so that contributions from computeForces are kept and accumulated through mappings once at the end
    addMBKv(b, (f_rayleighMass.getValue() == 0.0 ? 0.0 : -f_rayleighMass.getValue()), 0, h+f_rayleighStiffness.getValue(), false, true);

#else

    // compute the right-hand term of the equation system
    computeForce(b);             // b = f0

    //propagateDx(vel);            // dx = v
    //computeDf(f);                // f = df/dx v
    computeDfV(f);                // f = df/dx v
    b.peq(f,h+f_rayleighStiffness.getValue());      // b = f0 + (h+rs)df/dx v

    if (f_rayleighMass.getValue() != 0.0)
    {
        //f.clear();
        //addMdx(f,vel);
        //b.peq(f,-f_rayleighMass.getValue());     // b = f0 + (h+rs)df/dx v - rd M v
        //addMdx(b,VecId(),-f_rayleighMass.getValue()); // no need to propagate vel as dx again
        addMdx(b,vel,-f_rayleighMass.getValue()); // no need to propagate vel as dx again
    }

#endif

    b.teq(h);                           // b = h(f0 + (h+rs)df/dx v - rd M v)

    if( verbose )
        cerr<<"EulerImplicitSolver, f0 = "<< b <<endl;

    projectResponse(b);          // b is projected to the constrained space

    if( verbose )
        cerr<<"EulerImplicitSolver, projected f0 = "<< b <<endl;

    MultiMatrix matrix(this);
    matrix = MechanicalMatrix::K * (-h*(h+f_rayleighStiffness.getValue())) + MechanicalMatrix::M * (1+h*f_rayleighMass.getValue());

    //if( verbose )
//	cerr<<"EulerImplicitSolver, matrix = "<< (MechanicalMatrix::K * (-h*(h+f_rayleighStiffness.getValue())) + MechanicalMatrix::M * (1+h*f_rayleighMass.getValue())) << " = " << matrix <<endl;

    matrix.solve(x, b);
    // projectResponse(x);
    // x is the solution of the system

    // apply the solution
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    vel.peq( x );                       // vel = vel + x
    pos.peq( vel, h );                  // pos = pos + h vel
#else // single-operation optimization
    {
        simulation::MechanicalVMultiOpVisitor vmop;
        vmop.ops.resize(2);
        vmop.ops[0].first = (VecId)vel;
        vmop.ops[0].second.push_back(std::make_pair((VecId)vel,1.0));
        vmop.ops[0].second.push_back(std::make_pair((VecId)x,1.0));
        vmop.ops[1].first = (VecId)pos;
        vmop.ops[1].second.push_back(std::make_pair((VecId)pos,1.0));
        vmop.ops[1].second.push_back(std::make_pair((VecId)vel,h));
        vmop.execute(this->getContext());
    }
#endif
    if (f_velocityDamping.getValue()!=0.0)
        vel *= exp(-h*f_velocityDamping.getValue());

    if( verbose )
    {
        cerr<<"EulerImplicitSolver, final x = "<< pos <<endl;
        cerr<<"EulerImplicitSolver, final v = "<< vel <<endl;
    }
}

SOFA_DECL_CLASS(EulerImplicitSolver)

int EulerImplicitSolverClass = core::RegisterObject("Implicit time integrator using backward Euler scheme")
        .add< EulerImplicitSolver >()
        .addAlias("EulerImplicit");
;

} // namespace odesolver

} // namespace component

} // namespace sofa

