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
#include <SofaBaseLinearSolver/CGLinearSolver.inl>
#include <SofaDenseSolver/NewMatMatrix.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

int NewMatCGLinearSolverClass = core::RegisterObject("NewMat linear system solver using the conjugate gradient iterative algorithm")
        .add< CGLinearSolver< NewMatMatrix, NewMatVector > >()
        .add< CGLinearSolver< NewMatSymmetricMatrix, NewMatVector > >()
        .add< CGLinearSolver< NewMatBandMatrix, NewMatVector > >()
        .add< CGLinearSolver< NewMatSymmetricBandMatrix, NewMatVector > >()
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

