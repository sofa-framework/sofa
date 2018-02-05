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
#ifndef SOFA_COMPONENT_LINEARSOLVER_BLOCKJACOBIPRECONDITIONER_INL
#define SOFA_COMPONENT_LINEARSOLVER_BLOCKJACOBIPRECONDITIONER_INL

#include <SofaPreconditioner/BlockJacobiPreconditioner.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector>
BlockJacobiPreconditioner<TMatrix,TVector>::BlockJacobiPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
{
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    M.mul(z,r);
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::invert(Matrix& M)
{
    M.invert();
    if (f_verbose.getValue()) sout<<M<<sendl;
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
