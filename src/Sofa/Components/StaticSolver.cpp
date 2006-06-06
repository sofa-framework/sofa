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
}

void StaticSolver::solve(double)
{
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
    group->applyConstraints(b);         // b is projected to the constrained space

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
        }

// matrix-vector product
        group->propagateDx(p);          // dx = p
        group->computeDf(q);            // q = df/dx p
// filter the product to take the constraints into account
        group->applyConstraints(q);     // q is projected to the constrained space

        double den = p.dot(q);
        if( fabs(den)<smallDenominatorThreshold )
            break;
        alpha = rho/den;
        x.peq(p,alpha);                 // x = x + alpha p
        r.peq(q,-alpha);                // r = r - alpha r
        rho_1 = rho;
    }
// x is the solution of the system

// apply the solution
    pos.peq( x );
}

void create(StaticSolver*& obj, ObjectDescription* arg)
{
    obj = new StaticSolver();
    if (arg->getAttribute("iterations"))
        obj->maxCGIter = atoi(arg->getAttribute("iterations"));
    if (arg->getAttribute("threshold"))
        obj->smallDenominatorThreshold = atof(arg->getAttribute("threshold"));
}

SOFA_DECL_CLASS(StaticSolver)

Creator<ObjectFactory, StaticSolver> StaticSolverClass("Static");

} // namespace Components

} // namespace Sofa
