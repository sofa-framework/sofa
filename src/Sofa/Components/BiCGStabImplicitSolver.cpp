// Author: Jeremie Allard, Sim Group @ CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include "Sofa/Components/BiCGStabImplicitSolver.h"
//#include "Sofa/Core/IntegrationGroup.h"
#include "Sofa/Core/MultiVector.h"
#include "Common/ObjectFactory.h"

#include <math.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

BiCGStabImplicitSolver::BiCGStabImplicitSolver()
{
    maxCGIter = 25;
    smallDenominatorThreshold = 1e-5;
    tolerance = 1e-5;
    rayleighStiffness = 0.1;
}

BiCGStabImplicitSolver* BiCGStabImplicitSolver::setMaxIter( int n )
{
    maxCGIter = n;
    return this;
}

void BiCGStabImplicitSolver::solve(double dt)
{
    BiCGStabImplicitSolver* group = this;
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector dx(group, VecId::dx());
    MultiVector f(group, VecId::force());
    MultiVector b(group, V_DERIV);
    MultiVector p(group, V_DERIV);
    //MultiVector q(group, V_DERIV);
    MultiVector r(group, V_DERIV);
    MultiVector rtilde(group, V_DERIV);
    MultiVector s(group, V_DERIV);
    MultiVector t(group, V_DERIV);
    MultiVector x(group, V_DERIV);
    MultiVector v(group, V_DERIV);
    //MultiVector z(group, V_DERIV);
    double h = dt;

    if( getDebug() )
    {
        cerr<<"BiCGStabImplicitSolver, dt = "<< dt <<endl;
        cerr<<"BiCGStabImplicitSolver, initial x = "<< pos <<endl;
        cerr<<"BiCGStabImplicitSolver, initial v = "<< vel <<endl;
    }

    // compute the right-hand term of the equation system
    group->computeForce(b);             // b = f0
    if( getDebug() )
        cerr<<"BiCGStabImplicitSolver, f0 = "<< b <<endl;
    group->propagateDx(vel);            // dx = v
    group->computeDf(f);                // f = df/dx v
    b.peq(f,h);                         // b = f0+hdf/dx v
    b.teq(h);                           // b = h(f0+hdf/dx v)
    group->projectResponse(b);         // b is projected to the constrained space

    double normb = sqrt(b.dot(b));

    // -- solve the system using a bi conjugate gradient stabilized solution
    double rho_1=0, rho_2=0, alpha, beta, omega;
    group->v_clear( x );
    group->v_eq(r,b); // initial residual

    rtilde = r;

    if( getDebug() )
        cerr<<"BiCGStabImplicitSolver, r0 = "<< r <<endl;

    unsigned nb_iter;
    const char* endcond = "iterations";
    for( nb_iter=1; nb_iter<=maxCGIter; nb_iter++ )
    {
        //z = r; // no precond
        rho_1 = rtilde.dot(r);
        if( nb_iter==1 )
            p = r; //z;
        else
        {
            beta = ( rho_1 / rho_2 ) * ( alpha / omega );
            // p = r + beta * (p - omega * v);
            p.peq(v,-omega);
            p *= beta;
            p += r;
        }

        // matrix-vector product v = A * p
        group->propagateDx(p);          // dx = p
        group->computeDf(v);            // v = df/dx p
        v *= -h*(h+rayleighStiffness);  // v = -h(h+r) df/dx p
        group->addMdx( v, p);           // v = Mp -h(h+r) df/dx p
        // filter the product to take the constraints into account
        group->projectResponse(v);     // v is projected to the constrained space

        double den = rtilde.dot(v);
        if( fabs(den)<smallDenominatorThreshold )
        {
            endcond = "threshold1";
            break;
        }
        alpha = rho_1/den;
        s = r;
        s.peq(v,-alpha);                // s = r - alpha v
        x.peq(p,alpha);                 // x = x + alpha p

        double norms = sqrt(s.dot(s));
        if (norms / normb <= tolerance)
        {
            endcond = "tolerance1";
            break;
        }

        // matrix-vector product t = A * s
        group->propagateDx(s);          // dx = s
        group->computeDf(t);            // t = df/dx s
        t *= -h*(h+rayleighStiffness);  // t = -h(h+r) df/dx s
        group->addMdx( t, s);           // t = Ms -h(h+r) df/dx s
        // filter the product to take the constraints into account
        group->projectResponse(t);     // v is projected to the constrained space

        den = t.dot(t);
        if( fabs(den)<smallDenominatorThreshold )
        {
            endcond = "threshold2";
            break;
        }
        omega = t.dot(s)/den;

        r = s;
        r.peq(t,-omega);                // r = s - omega t
        x.peq(s,omega);                 // x = x + omega s

        double normr = sqrt(r.dot(r));
        if (normr / normb <= tolerance)
        {
            endcond = "tolerance2";
            break;
        }

        rho_2 = rho_1;
    }
    // x is the solution of the system
    cerr<<"BiCGStabImplicitSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;

    // apply the solution
    vel.peq( x );                       // vel = vel + x
    pos.peq( vel, h );                  // pos = pos + h vel

    if( getDebug() )
    {
        cerr<<"BiCGStabImplicitSolver, final x = "<< pos <<endl;
        cerr<<"BiCGStabImplicitSolver, final v = "<< vel <<endl;
    }
}

void create(BiCGStabImplicitSolver*& obj, ObjectDescription* arg)
{
    obj = new BiCGStabImplicitSolver();
    if (arg->getAttribute("iterations"))
        obj->setMaxIter( atoi(arg->getAttribute("iterations")) );
    if (arg->getAttribute("threshold"))
        obj->smallDenominatorThreshold = atof(arg->getAttribute("threshold"));
    if (arg->getAttribute("tolerance"))
        obj->tolerance = atof(arg->getAttribute("tolerance"));
    if (arg->getAttribute("stiffness"))
        obj->rayleighStiffness = atof(arg->getAttribute("stiffness"));
    //if (arg->getAttribute("debug"))
    //    obj->setDebug( atoi(arg->getAttribute("debug"))!=0 );
}

SOFA_DECL_CLASS(BiCGStabImplicit)

Creator<ObjectFactory, BiCGStabImplicitSolver> BiCGStabImplicitSolverClass("BiCGStabImplicit");

} // namespace Components

} // namespace Sofa
