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

#include <sofa/component/linearsolver/iterative/LinearSolverTask.h>

namespace sofa::component::linearsolver
{
// Force template instantiation
using namespace sofa::linearalgebra;

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< GraphScatteredMatrix, GraphScatteredVector >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< FullMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< SparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< DiagonalMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< BlockDiagonalMatrix<3,SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ComputeColumnTask< RotationMatrix<SReal>, FullVector<SReal> >;

}//namespace lineasolver