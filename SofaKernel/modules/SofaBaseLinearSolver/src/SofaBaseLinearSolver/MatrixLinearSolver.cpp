/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP
#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.inl>

namespace sofa::component::linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< FullMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< FullMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< SparseMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< SparseMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< DiagonalMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< DiagonalMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< BlockDiagonalMatrix<3,double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< RotationMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< RotationMatrix<float>, FullVector<float>, NoThreadManager >;

} // namespace sofa::component::linearsolver
