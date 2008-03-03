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
{
}

void StaticSolver::solve(double dt)
{
    MultiVector b(this, VecId::V_DERIV);
    MultiVector x(this, VecId::V_DERIV);
    MultiVector pos(this, VecId::position());

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    // compute the right-hand term of the equation system
    this->computeForce(b);             // b = f0
    this->projectResponse(b);         // b is projected to the constrained space
    b.teq(-1);

    if( f_printLog.getValue() )
        cerr<<"StaticSolver, f0 = "<< b <<endl;
    MultiMatrix matrix(this);
    matrix = MechanicalMatrix::K;

    if( f_printLog.getValue() )
        cerr<<"StaticSolver, matrix = "<< (MechanicalMatrix::K) << " = " << matrix <<endl;

    matrix.solve(x,b);
    // x is the solution of the system

    // apply the solution
    /*    cerr<<"StaticSolver::solve, nb iter = "<<nb_iter<<endl;
     cerr<<"StaticSolver::solve, solution = "<<x<<endl;*/

    if( f_printLog.getValue() )
        cerr<<"StaticSolver, solution = "<< x <<endl;
    pos.peq( x );
    /*    cerr<<"StaticSolver::solve, new pos = "<<pos<<endl;*/
}

int StaticSolverClass = core::RegisterObject("A solver which seeks the static equilibrium of the scene it monitors")
        .add< StaticSolver >();

SOFA_DECL_CLASS(StaticSolver)


} // namespace odesolver

} // namespace component

} // namespace sofa

