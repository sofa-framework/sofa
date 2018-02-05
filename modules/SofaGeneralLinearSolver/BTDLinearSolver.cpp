/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_CPP

#include <SofaGeneralLinearSolver/BTDLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>

namespace sofa
{

namespace component
{

namespace linearsolver
{

SOFA_DECL_CLASS(BTDLinearSolver)

int BTDLinearSolverClass = core::RegisterObject("Linear system solver using Thomas Algorithm for Block Tridiagonal matrices")
#ifndef SOFA_FLOAT
.add< BTDLinearSolver<BTDMatrix<6,double>,BlockVector<6,double> > >(true)
#endif
#ifndef SOFA_DOUBLE
.add< BTDLinearSolver<BTDMatrix<6,float>,BlockVector<6,float> > >()
#endif
//.add< BTDLinearSolver<BTDMatrix<3,double>,BlockVector<3,double> > >()
//.add< BTDLinearSolver<BTDMatrix<3,float>,BlockVector<3,float> > >()
//.add< BTDLinearSolver<BTDMatrix<2,double>,BlockVector<2,double> > >()
//.add< BTDLinearSolver<BTDMatrix<2,float>,BlockVector<2,float> > >()
//.add< BTDLinearSolver<BTDMatrix<1,double>,BlockVector<1,double> > >()
//.add< BTDLinearSolver<BTDMatrix<1,float>,BlockVector<1,float> > >()
//.add< BTDLinearSolver<NewMatMatrix,NewMatVector> >()
//.add< BTDLinearSolver<NewMatSymmetricMatrix,NewMatVector> >()
//.add< BTDLinearSolver<NewMatBandMatrix,NewMatVector> >(true)
//.add< BTDLinearSolver<NewMatSymmetricBandMatrix,NewMatVector> >()
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_LINEAR_SOLVER_API BTDLinearSolver< BTDMatrix<6, double>, BlockVector<6, double> >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_LINEAR_SOLVER_API BTDLinearSolver< BTDMatrix<6, float>, BlockVector<6, float> >;
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

