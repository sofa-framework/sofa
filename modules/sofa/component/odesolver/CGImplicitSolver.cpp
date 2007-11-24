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
#include <sofa/component/odesolver/CGImplicitSolver.h>
//#include "Sofa/Core/IntegrationGroup.h"
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

CGImplicitSolver::CGImplicitSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
    , f_velocityDamping( initData(&f_velocityDamping,0.,"vdamping","Velocity decay coefficient (no decay if null)") )
{
    //     maxCGIter = 25;
    //     smallDenominatorThreshold = 1e-5;
    //     tolerance = 1e-5;
    //     rayleighStiffness = 0.1;
    //     rayleighMass = 0.1;
    //     velocityDamping = 0;

}

// CGImplicitSolver* CGImplicitSolver::setMaxIter( int n )
// {
//     maxCGIter = n;
//     return this;
// }

void CGImplicitSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector f(this, VecId::force());
    MultiVector b(this, VecId::V_DERIV);
    MultiVector p(this, VecId::V_DERIV);
    MultiVector q(this, VecId::V_DERIV);
    //MultiVector q2(this, VecId::V_DERIV);
    //MultiVector r(this, VecId::V_DERIV);
    MultiVector x(this, VecId::V_DERIV);

    double h = dt;
    bool printLog = f_printLog.getValue();


    projectResponse(vel);          // initial velocities are projected to the constrained space

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


    b.teq(h);                           // b = h(f0 + (h+rs)df/dx v - rd M v)
    projectResponse(b);          // b is projected to the constrained space

    double normb2 = b.dot(b);
    double normb = sqrt(normb2);


    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;

    v_clear( x );
    //v_eq(r,b); // initial residual
    MultiVector& r = b; // b is never used after this point

    /*if( printLog )
    {
        cerr<<"CGImplicitSolver, dt = "<< dt <<endl;
        cerr<<"CGImplicitSolver, initial x = "<< pos <<endl;
        cerr<<"CGImplicitSolver, initial v = "<< vel <<endl;
        cerr<<"CGImplicitSolver, f0 = "<< b <<endl;
        cerr<<"CGImplicitSolver, r0 = "<< r <<endl;
    }*/

    unsigned nb_iter;
    const char* endcond = "iterations";
    for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
    {

// 		printWithElapsedTime( x, helper::system::thread::CTime::getTime()-time0,std::cout );

        //z = r; // no precond
        //rho = r.dot(z);
        rho = (nb_iter==1) ? normb2 : r.dot(r);

        if (nb_iter>1)
        {
            double normr = sqrt(rho); //sqrt(r.dot(r));
            if (normr/normb <= f_tolerance.getValue())
            {
                endcond = "tolerance";
                break;
            }
        }

        if( nb_iter==1 )
            p = r; //z;
        else
        {
            beta = rho / rho_1;
            //p *= beta;
            //p += r; //z;
            v_op(p,r,p,beta); // p = p*beta + r
        }

        /*if( printLog )
        {
            cerr<<"p : "<<p<<endl;
        }*/

        // matrix-vector product
        propagateDx(p);          // dx = p
        computeDf(q);            // q = df/dx p

        /*if( printLog )
        {
            cerr<<"q = df/dx p : "<<q<<endl;
        }*/

        q *= -h*(h+f_rayleighStiffness.getValue());  // q = -h(h+rs) df/dx p

        /*if( printLog )
        {
            cerr<<"q = -h(h+rs) df/dx p : "<<q<<endl;
        }*/
        //
        // 		cerr<<"-h(h+rs) df/dx p : "<<q<<endl;
        // 		cerr<<"f_rayleighMass.getValue() : "<<f_rayleighMass.getValue()<<endl;

        // apply global Rayleigh damping
        if (f_rayleighMass.getValue()==0.0)
        {
            //addMdx( q, p);           // q = Mp -h(h+rs) df/dx p
            addMdx(q); // no need to propagate p as dx again
        }
        else
        {
            //q2.clear();
            //addMdx( q2, p);
            //q.peq(q2,(1+h*f_rayleighMass.getValue())); // q = Mp -h(h+rs) df/dx p +hr Mp  =  (M + dt(rd M + rs K) + dt2 K) dx
            addMdx(q,VecId(),(1+h*f_rayleighMass.getValue())); // no need to propagate p as dx again
        }
        /*if( printLog )
        {
            cerr<<"q = Mp -h(h+rs) df/dx p +hr Mp  =  "<<q<<endl;
        }*/

        // filter the product to take the constraints into account
        //
        projectResponse(q);     // q is projected to the constrained space
        /*if( printLog )
        {
            cerr<<"q after constraint projection : "<<q<<endl;
        }*/

        double den = p.dot(q);


        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            endcond = "threshold";
            if( printLog )
            {
                //                 cerr<<"CGImplicitSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<endl;
            }
            break;
        }
        alpha = rho/den;
        x.peq(p,alpha);                 // x = x + alpha p
        r.peq(q,-alpha);                // r = r - alpha q
        /*if( printLog ){
            cerr<<"den = "<<den<<", alpha = "<<alpha<<endl;
            cerr<<"x : "<<x<<endl;
            cerr<<"r : "<<r<<endl;
        }*/

        rho_1 = rho;
    }
    // x is the solution of the system

    // apply the solution
    vel.peq( x );                       // vel = vel + x
    pos.peq( vel, h );                  // pos = pos + h vel
    if (f_velocityDamping.getValue()!=0.0)
        vel *= exp(-h*f_velocityDamping.getValue());

    if( printLog )
    {
        cerr<<"CGImplicitSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;
        //cerr<<"CGImplicitSolver::solve, solution = "<<x<<endl;
        //cerr<<"CGImplicitSolver, final x = "<< pos <<endl;
        //cerr<<"CGImplicitSolver, final v = "<< vel <<endl;
    }
}

SOFA_DECL_CLASS(CGImplicit)

int CGImplicitSolverClass = core::RegisterObject("Implicit time integration using the filtered conjugate gradient")
        .add< CGImplicitSolver >()
        .addAlias("CGImplicit");
;

} // namespace odesolver

} // namespace component

} // namespace sofa

