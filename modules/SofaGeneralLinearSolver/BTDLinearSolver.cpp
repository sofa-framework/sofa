/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

static int BTDLinearSolverClass = core::RegisterObject("Linear system solver using Thomas Algorithm for Block Tridiagonal matrices")
    .add< BTDLinearSolver<BTDMatrix<6,double>,BlockVector<6,double> > >(true)
;

template class SOFA_GENERAL_LINEAR_SOLVER_API BTDLinearSolver< BTDMatrix<6, double>, BlockVector<6, double> >;

} // namespace linearsolver

} // namespace component

} // namespace sofa

