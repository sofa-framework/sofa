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
#pragma once

#include <Eigen/Sparse>
#include <sofa/linearalgebra/SparseMatrixProduct.h>

namespace sofa::linearalgebra
{

template<> void SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float> >::computeRegularProduct();
template<> void SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double> >::computeRegularProduct();
template<> void SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >::computeRegularProduct();
template<> void SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >::computeRegularProduct();

#if !defined(SOFA_LINEARAGEBRA_SPARSEMATRIXPRODUCT_EIGENSPARSEMATRIX_CPP)
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float> >;
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double> >;

    extern template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<float, Eigen::RowMajor> >;
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixProduct<Eigen::SparseMatrix<double, Eigen::RowMajor> >;
#endif

} //namespace sofa::linearalgebra