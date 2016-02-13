/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaBaseLinearSolver/BTDLinearSolver.inl>
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

} // namespace linearsolver

} // namespace component

} // namespace sofa

