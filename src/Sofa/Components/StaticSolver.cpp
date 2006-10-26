// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include "Sofa/Components/StaticSolver.h"
#include "Sofa/Core/MultiVector.h"
#include "Common/ObjectFactory.h"

#include <math.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

StaticSolver::StaticSolver()
{
    maxCGIter = 25;
    smallDenominatorThreshold = 1e-5;
    /*    newField(&maxCGIter,"iterations","maximum number of iterations of the Conjugate Gradient solution");
        newField(&smallDenominatorThreshold,"threshold","minimum value of the denominator in the conjugate Gradient solution");*/
}

void StaticSolver::solve(double)
{
    //Abstract::BaseContext* group = getContext();
    OdeSolver* group = this;
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector dx(group, VecId::dx());
    MultiVector f(group, VecId::force());
    MultiVector b(group, V_DERIV);
    MultiVector p(group, V_DERIV);
    MultiVector q(group, V_DERIV);
    MultiVector r(group, V_DERIV);
    MultiVector x(group, V_DERIV);
    MultiVector z(group, V_DERIV);

    // compute the right-hand term of the equation system
    group->computeForce(b);             // b = f0
    b.teq(-1);                          // b = -f0
    group->projectResponse(b);         // b is projected to the constrained space
    //     cerr<<"StaticSolver::solve, initial position = "<<pos<<endl;
    //     cerr<<"StaticSolver::solve, b = "<<b<<endl;

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;
    group->v_clear( x );
    group->v_eq(r,b); // initial residual

    unsigned nb_iter;
    for( nb_iter=1; nb_iter<=maxCGIter; nb_iter++ )
    {
        z = r; // no precond
        rho = r.dot(z);
        if( nb_iter==1 )
            p = z;
        else
        {
            beta = rho / rho_1;
            p *= beta;
            p += z;
            //cerr<<"StaticSolver::solve, new p = "<< p <<endl;
        }

        // matrix-vector product
        /*        cerr<<"StaticSolver::solve, dx = "<<p<<endl;*/
        group->propagateDx(p);          // dx = p
        group->computeDf(q);            // q = df/dx p
        /*        cerr<<"StaticSolver::solve, df = "<<q<<endl;*/
        // filter the product to take the constraints into account
        group->projectResponse(q);     // q is projected to the constrained space
        //         cerr<<"StaticSolver::solve, df filtered = "<<q<<endl;

        double den = p.dot(q);
        /*        cerr<<"StaticSolver::solve, den = "<<den<<endl;*/
        if( fabs(den)<smallDenominatorThreshold )
            break;
        alpha = rho/den;
        /*        cerr<<"StaticSolver::solve, rho = "<< rho <<endl;
                cerr<<"StaticSolver::solve, den = "<< den <<endl;
                cerr<<"StaticSolver::solve, alpha = "<< alpha <<endl;*/
        x.peq(p,alpha);                 // x = x + alpha p
        r.peq(q,-alpha);                // r = r - alpha r
        rho_1 = rho;
        /*        cerr<<"StaticSolver::solve, x = "<<x<<endl;
                cerr<<"StaticSolver::solve, r = "<<r<<endl;
                cerr<<"StaticSolver::solve, residual = "<<sqrt(r.dot(r))<<endl;*/
    }
    // x is the solution of the system

    // apply the solution
    /*    cerr<<"StaticSolver::solve, nb iter = "<<nb_iter<<endl;
        cerr<<"StaticSolver::solve, solution = "<<x<<endl;*/
    pos.peq( x );
    /*    cerr<<"StaticSolver::solve, new pos = "<<pos<<endl;*/
}

void create(StaticSolver*& obj, ObjectDescription* arg)
{
    obj = new StaticSolver();
    obj->parseFields( arg->getAttributeMap() );
}

SOFA_DECL_CLASS(StaticSolver)

Creator<ObjectFactory, StaticSolver> StaticSolverClass("Static");

} // namespace Components

} // namespace Sofa

