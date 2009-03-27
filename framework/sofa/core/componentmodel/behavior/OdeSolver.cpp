/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

OdeSolver::OdeSolver()
#ifdef SOFA_HAVE_EIGEN2
    :
    constraintAcc( initData( &constraintAcc, false, "constraintAcc", "Constraint the acceleration")),
    constraintVel( initData( &constraintVel, false, "constraintVel", "Constraint the velocity")),
    constraintPos( initData( &constraintPos, false, "constraintPos", "Constraint the position")),
    constraintResolution( initData( &constraintResolution, false, "constraintResolution", "Using Gauss-Seidel to solve the constraint.\nOtherwise, use direct LU resolution.")),
    numIterations( initData( &numIterations, (unsigned int)25, "numIterations", "Number of iterations for Gauss-Seidel when solving the Constraints")),
    maxError( initData( &maxError, 0.0000001, "maxError", "Max error for Gauss-Seidel algorithm when solving the constraints"))
#endif
{}

OdeSolver::~OdeSolver()
{}

//const OdeSolver::MechanicalMatrix OdeSolver::M(1,0,0);
//const OdeSolver::MechanicalMatrix OdeSolver::B(0,1,0);
//const OdeSolver::MechanicalMatrix OdeSolver::K(0,0,1);

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

