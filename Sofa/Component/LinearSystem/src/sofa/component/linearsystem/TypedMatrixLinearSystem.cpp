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

#include <sofa/component/linearsystem/TypedMatrixLinearSystem.inl>

#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>

namespace sofa::component::linearsystem
{

using namespace sofa::linearalgebra;

template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< FullMatrix<double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< FullMatrix<float>, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< SparseMatrix<double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< SparseMatrix<float>, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<float>, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,double> >, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,float> >, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,double> >, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,float> >, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,double> >, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,float> >, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,double> >, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,float> >, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,double> >, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,float> >, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< DiagonalMatrix<double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< DiagonalMatrix<float>, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< BlockDiagonalMatrix<3,double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< BlockDiagonalMatrix<3,float>, FullVector<float> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< RotationMatrix<double>, FullVector<double> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< RotationMatrix<float>, FullVector<float> >;

} //namespace sofa::component::linearsystem
