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
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>

using std::cout;
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

StaticSolver::StaticSolver()
    : f_maxCGIter( initData(&f_maxCGIter,(unsigned)25,"iterations","Maximum number of iterations for the conjugated gradient algorithmIndices of the fixed points") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
{
}

void StaticSolver::solve(double , VecId b)
{
    /*std::cout << "Static Solver will solve knowing b!! "<< this->getName() << "\n";*/
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector dx(this, VecId::dx());
    MultiVector f(this, VecId::force());
    MultiVector p(this, VecId::V_DERIV);
    MultiVector q(this, VecId::V_DERIV);
    MultiVector r(this, VecId::V_DERIV);
    MultiVector x(this, VecId::V_DERIV);
    MultiVector z(this, VecId::V_DERIV);

//    b.teq(-1);                          // b = -f0
    this->projectResponse(b);         // b is projected to the constrained space
    //     cerr<<"StaticSolver::solve, initial position = "<<pos<<endl;
    //     cerr<<"StaticSolver::solve, b = "<<b<<endl;

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;
    this->v_clear( x );
    this->v_eq(r,b); // initial residual

    unsigned nb_iter;
    for( nb_iter=1; nb_iter<=f_maxCGIter.getValue(); nb_iter++ )
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
        this->propagateDx(p);          // dx = p
        this->computeDf(q);            // q = df/dx p
        /*        cerr<<"StaticSolver::solve, df = "<<q<<endl;*/
        // filter the product to take the constraints into account
        this->projectResponse(q);     // q is projected to the constrained space
        //         cerr<<"StaticSolver::solve, df filtered = "<<q<<endl;

        double den = p.dot(q);
        /*        cerr<<"StaticSolver::solve, den = "<<den<<endl;*/
        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            //std::cout << "nb_iter = " << nb_iter << std::endl;
            break;
        }

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
void StaticSolver::solve(double dt)
{
    MultiVector b(this, VecId::V_DERIV);

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    // compute the right-hand term of the equation system
    this->computeForce(b);             // b = f0
    b.teq(-1);
    solve(dt, b);
}

int StaticSolverClass = core::RegisterObject("A solver which seeks the static equilibrium of the scene it monitors")
        .add< StaticSolver >();

SOFA_DECL_CLASS(StaticSolver)


} // namespace odesolver

} // namespace component

} // namespace sofa

