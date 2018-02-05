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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaSparseSolver/SparseLUSolver.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

SOFA_DECL_CLASS(SparseLUSolver)

int SparseLUSolverClass = core::RegisterObject("Direct linear solver based on Sparse LU factorization, implemented with the CSPARSE library")
        .add< SparseLUSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >()
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

