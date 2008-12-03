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
    : massCoef( initData(&massCoef,(double)0.0,"massCoef","coefficient associated with the mass matrix in the equation system") )
    , dampingCoef( initData(&dampingCoef,(double)0.0,"dampingCoef","coefficient associated with the mass matrix in the equation system") )
    , stiffnessCoef( initData(&stiffnessCoef,(double)1.0,"stiffnessCoef","coefficient associated with the mass matrix in the equation system") )
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
    //matrix = MechanicalMatrix::K;
    matrix = MechanicalMatrix(massCoef.getValue(),dampingCoef.getValue(),stiffnessCoef.getValue());

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


#ifdef SOFA_HAVE_LAPACK
    applyConstraints();
#endif
    /*    cerr<<"StaticSolver::solve, new pos = "<<pos<<endl;*/
}

int StaticSolverClass = core::RegisterObject("A solver which seeks the static equilibrium of the scene it monitors")
        .add< StaticSolver >();

SOFA_DECL_CLASS(StaticSolver)


} // namespace odesolver

} // namespace component

} // namespace sofa

