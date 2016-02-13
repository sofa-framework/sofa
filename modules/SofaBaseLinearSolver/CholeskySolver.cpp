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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#define SOFA_COMPONENT_LINEARSOLVER_CHOLESKYSOLVER_CPP
#include <SofaBaseLinearSolver/CholeskySolver.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(CholeskySolver)

int CholeskySolverClass = core::RegisterObject("Direct linear solver based on Cholesky factorization, for dense matrices")
#ifndef SOFA_FLOAT
        .add< CholeskySolver< SparseMatrix<double>, FullVector<double> > >(true)
        .add< CholeskySolver< FullMatrix<double>, FullVector<double> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CholeskySolver< SparseMatrix<float>, FullVector<float> > >(true)
        .add< CholeskySolver< FullMatrix<float>, FullVector<float> > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_LINEAR_SOLVER_API CholeskySolver< SparseMatrix<double>, FullVector<double> >;
template class SOFA_BASE_LINEAR_SOLVER_API CholeskySolver< FullMatrix<double>, FullVector<double> >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_LINEAR_SOLVER_API CholeskySolver< SparseMatrix<float>, FullVector<float> >;
template class SOFA_BASE_LINEAR_SOLVER_API CholeskySolver< FullMatrix<float>, FullVector<float> >;
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa
