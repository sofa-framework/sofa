// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include "Sofa/Components/CGImplicitSolver.h"
#include "Sofa/Core/IntegrationGroup.h"
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

CGImplicitSolver::CGImplicitSolver()
{
    maxCGIter = 25;
    smallDenominatorThreshold = 1e-5;
    tolerance = 1e-5;
    rayleighStiffness = 0.1;
    rayleighDamping = 0.1;
}

CGImplicitSolver* CGImplicitSolver::setMaxIter( int n )
{
    maxCGIter = n;
    return this;
}

void CGImplicitSolver::solve(double dt)
{
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector dx(group, VecId::dx());
    MultiVector f(group, VecId::force());
    MultiVector b(group, V_DERIV);
    MultiVector p(group, V_DERIV);
    MultiVector q(group, V_DERIV);
    MultiVector q2(group, V_DERIV);
    MultiVector r(group, V_DERIV);
    MultiVector x(group, V_DERIV);
    //MultiVector z(group, V_DERIV);
    double h = dt;

    if( getDebug() )
    {
        cerr<<"CGImplicitSolver, dt = "<< dt <<endl;
        cerr<<"CGImplicitSolver, initial x = "<< pos <<endl;
        cerr<<"CGImplicitSolver, initial v = "<< vel <<endl;
    }

    // compute the right-hand term of the equation system
    group->computeForce(b);             // b = f0
    if( getDebug() )
        cerr<<"CGImplicitSolver, f0 = "<< b <<endl;
    group->propagateDx(vel);            // dx = v
    group->computeDf(f);                // f = df/dx v
    b.peq(f,h+rayleighStiffness);      // b = f0 + (h+rs)df/dx v
    if (rayleighDamping != 0.0)
    {
        f.clear();
        group->addMdx(f,dx);
        b.peq(f,-rayleighDamping);     // b = f0 + (h+rs)df/dx v - rd M v
    }
    b.teq(h);                           // b = h(f0 + (h+rs)df/dx v - rd M v)
    group->projectResponse(b);          // b is projected to the constrained space

    double normb = sqrt(b.dot(b));

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;
    group->v_clear( x );
    group->v_eq(r,b); // initial residual

    if( getDebug() )
        cerr<<"CGImplicitSolver, r0 = "<< r <<endl;

    unsigned nb_iter;
    const char* endcond = "iterations";
    for( nb_iter=1; nb_iter<=maxCGIter; nb_iter++ )
    {
        //z = r; // no precond
        //rho = r.dot(z);
        rho = r.dot(r);
        if( nb_iter==1 )
            p = r; //z;
        else
        {
            beta = rho / rho_1;
            p *= beta;
            p += r; //z;
        }

        // matrix-vector product
        group->propagateDx(p);          // dx = p
        group->computeDf(q);            // q = df/dx p
        q *= -h*(h+rayleighStiffness);  // q = -h(h+rs) df/dx p
        if (rayleighDamping==0.0)
            group->addMdx( q, p);           // q = Mp -h(h+rs) df/dx p
        else
        {
            q2.clear();
            group->addMdx( q2, p);
            q.peq(q2,(1+h*rayleighDamping)); // q = Mp -h(h+rs) df/dx p +hr Mp  =  (M + dt(rd M + rs K) + dt2 K) dx
        }
        // filter the product to take the constraints into account
        group->projectResponse(q);     // q is projected to the constrained space

        double den = p.dot(q);
        if( fabs(den)<smallDenominatorThreshold )
        {
            endcond = "threshold";
            break;
        }
        alpha = rho/den;
        x.peq(p,alpha);                 // x = x + alpha p
        r.peq(q,-alpha);                // r = r - alpha r

        double normr = sqrt(r.dot(r));
        if (normr/normb <= tolerance)
        {
            endcond = "tolerance";
            break;
        }
        rho_1 = rho;
    }
    // x is the solution of the system
    //cerr<<"CGImplicitSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;

    // apply the solution
    vel.peq( x );                       // vel = vel + x
    pos.peq( vel, h );                  // pos = pos + h vel

    if( getDebug() )
    {
        cerr<<"CGImplicitSolver, final x = "<< pos <<endl;
        cerr<<"CGImplicitSolver, final v = "<< vel <<endl;
    }
}

void create(CGImplicitSolver*& obj, ObjectDescription* arg)
{
    obj = new CGImplicitSolver();
    if (arg->getAttribute("iterations"))
        obj->setMaxIter( atoi(arg->getAttribute("iterations")) );
    if (arg->getAttribute("threshold"))
        obj->smallDenominatorThreshold = atof(arg->getAttribute("threshold"));
    if (arg->getAttribute("tolerance"))
        obj->tolerance = atof(arg->getAttribute("tolerance"));
    if (arg->getAttribute("stiffness"))
        obj->rayleighStiffness = atof(arg->getAttribute("stiffness"));
    if (arg->getAttribute("damping"))
        obj->rayleighDamping = atof(arg->getAttribute("damping"));
    if (arg->getAttribute("debug"))
        obj->setDebug( atoi(arg->getAttribute("debug"))!=0 );
}

SOFA_DECL_CLASS(CGImplicit)

Creator<ObjectFactory, CGImplicitSolver> CGImplicitSolverClass("CGImplicit");

} // namespace Components

} // namespace Sofa
