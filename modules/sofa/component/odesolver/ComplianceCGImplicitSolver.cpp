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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include <sofa/component/odesolver/ComplianceCGImplicitSolver.h>
#include <sofa/core/visual/VisualParams.h>
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

ComplianceCGImplicitSolver::ComplianceCGImplicitSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
    , f_velocityDamping( initData(&f_velocityDamping,0.,"vdamping","Velocity decay coefficient (no decay if null)") )
{
    firstCallToSolve = true;
}


void ComplianceCGImplicitSolver::solve(double dt)
{
    MultiVector pos(this, core::VecCoordId::position());
    MultiVector vel(this, core::VecDerivId::velocity());
    MultiVector freePos(this, core::VecCoordId::freePosition());
    MultiVector freeVel(this, core::VecDerivId::freeVelocity());
    MultiVector f (this, core::VecDerivId::force());
    MultiVector b (this, core::V_DERIV);
    MultiVector p (this, core::V_DERIV);
    MultiVector q (this, core::V_DERIV);
    MultiVector q2(this, core::V_DERIV);
    MultiVector r (this, core::V_DERIV);
    MultiVector x (this, core::V_DERIV);

    if (!firstCallToSolve)
    {
        vel.eq(freeVel);
        pos.eq(freePos);
    }
    else
    {
        freeVel.eq(vel);
        freePos.eq(pos);
    }

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    firstCallToSolve = !firstCallToSolve;

    double h = dt;
    bool printLog = f_printLog.getValue();


    projectResponse(vel);          // initial velocities are projected to the constrained space

    // compute the right-hand term of the equation system
    computeForce(b);             // b = f0
    propagateDx(vel);            // dx = v
    computeDf(f);                // f = df/dx v
    b.peq(f,h+f_rayleighStiffness.getValue());      // b = f0 + (h+rs)df/dx v


    if (f_rayleighMass.getValue() != 0.0)
    {
        f.clear();
        addMdx(f,vel);
        b.peq(f,-f_rayleighMass.getValue());     // b = f0 + (h+rs)df/dx v - rd M v
    }


    b.teq(h);                           // b = h(f0 + (h+rs)df/dx v - rd M v)
    projectResponse(b);          // b is projected to the constrained space

    double normb = sqrt(b.dot(b));


    // -- solve the system using a conjugate gradient solution
    double rho_1=0, beta;
    v_clear( x );
    v_eq(r,b); // initial residual

    if( printLog )
    {
        serr<<"ComplianceCGImplicitSolver, dt = "<< dt <<sendl;
        serr<<"ComplianceCGImplicitSolver, initial x = "<< pos <<sendl;
        serr<<"ComplianceCGImplicitSolver, initial v = "<< vel <<sendl;
        serr<<"ComplianceCGImplicitSolver, f0 = "<< b <<sendl;
        serr<<"ComplianceCGImplicitSolver, r0 = "<< r <<sendl;
    }

    unsigned nb_iter;
    const char* endcond = "iterations";
    for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
    {

        //z = r; // no precond
        //double rho = r.dot(z);
        double rho = r.dot(r);


        if( nb_iter==1 )
            p = r; //z;
        else
        {
            beta = rho / rho_1;
            p *= beta;
            p += r; //z;
        }

        if( printLog )
        {
            serr<<"p : "<<p<<sendl;
        }

        // matrix-vector product
        propagateDx(p);          // dx = p
        computeDf(q);            // q = df/dx p

        if( printLog )
        {
            serr<<"q = df/dx p : "<<q<<sendl;
        }

        q *= -h*(h+f_rayleighStiffness.getValue());  // q = -h(h+rs) df/dx p

        if( printLog )
        {
            serr<<"q = -h(h+rs) df/dx p : "<<q<<sendl;
        }
        //
        // 		serr<<"-h(h+rs) df/dx p : "<<q<<sendl;
        // 		serr<<"f_rayleighMass.getValue() : "<<f_rayleighMass.getValue()<<sendl;

        // apply global Rayleigh damping
        if (f_rayleighMass.getValue()==0.0)
            addMdx( q, p);           // q = Mp -h(h+rs) df/dx p
        else
        {
            q2.clear();
            addMdx( q2, p);
            q.peq(q2,(1+h*f_rayleighMass.getValue())); // q = Mp -h(h+rs) df/dx p +hr Mp  =  (M + dt(rd M + rs K) + dt2 K) dx
        }
        if( printLog )
        {
            serr<<"q = Mp -h(h+rs) df/dx p +hr Mp  =  "<<q<<sendl;
        }

        // filter the product to take the constraints into account
        //
        projectResponse(q);     // q is projected to the constrained space
        if( printLog )
        {
            serr<<"q after constraint projection : "<<q<<sendl;
        }

        double den = p.dot(q);


        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            endcond = "threshold";
            if( printLog )
            {
                //                 serr<<"CGImplicitSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<sendl;
            }
            break;
        }
        double alpha = rho/den;
        x.peq(p,alpha);                 // x = x + alpha p
        r.peq(q,-alpha);                // r = r - alpha r
        if( printLog )
        {
            serr<<"den = "<<den<<", alpha = "<<alpha<<sendl;
            serr<<"x : "<<x<<sendl;
            serr<<"r : "<<r<<sendl;
        }


        double normr = sqrt(r.dot(r));
        if (normr/normb <= f_tolerance.getValue())
        {
            endcond = "tolerance";
            break;
        }
        rho_1 = rho;
    }
    // x is the solution of the system

    // apply the solution
    vel.peq( x );                       // vel = vel + x
    solveConstraint(dt,vel, core::behavior::BaseConstraintSet::VEL);
    pos.peq( vel, h );                  // pos = pos + h vel
    solveConstraint(dt,pos, core::behavior::BaseConstraintSet::POS);
    if (f_velocityDamping.getValue()!=0.0)
        vel *= exp(-h*f_velocityDamping.getValue());

    if( printLog )
    {
        serr<<"ComplianceCGImplicitSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<sendl;
        serr<<"ComplianceCGImplicitSolver::solve, solution = "<<x<<sendl;
        serr<<"ComplianceCGImplicitSolver, final x = "<< pos <<sendl;
        serr<<"ComplianceCGImplicitSolver, final v = "<< vel <<sendl;
    }
}


SOFA_DECL_CLASS(ComplianceCGImplicitSolver)

int ComplianceCGImplicitSolverClass = core::RegisterObject("Implicit time integration using the filtered conjugate gradient")
        .add< ComplianceCGImplicitSolver >()
        ;


} // namespace odesolver

} // namespace component

} // namespace sofa
