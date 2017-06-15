/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_LINEARSOLVER_SPARSELDLSOLVER_CPP
#include <SofaSparseSolver/SparseLDLSolver.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

SOFA_DECL_CLASS(SparseLDLSolver)

int SparseLDLSolverClass = core::RegisterObject("Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library")
        .add< SparseLDLSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        .add< SparseLDLSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >,FullVector<double> > >(true)
        .add< SparseLDLSolver< CompressedRowSparseMatrix<float>,FullVector<float> > >(true)
        .add< SparseLDLSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >,FullVector<float> > >(true)
        ;

template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix<double>,FullVector<double> >;
template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,double> >,FullVector<double> >;
template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix<float>,FullVector<float> >;
template class SOFA_SPARSE_SOLVER_API SparseLDLSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,float> >,FullVector<float> >;

} // namespace linearsolver

} // namespace component

} // namespace sofa
