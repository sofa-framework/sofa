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
#include <sofa/linearalgebra/SparseMatrixStorageOrder.h>
#include <sofa/linearalgebra/config.h>

namespace sofa::linearalgebra
{

/**
 * Single-precision float + Column-major sparse matrix representation
 */
///@{
template<> float SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::value() const;
template<> Eigen::SparseMatrix<float>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::row() const;
template<> Eigen::SparseMatrix<float>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >::InnerIterator::col() const;
///@}

/**
 * Double-precision float + Column-major sparse matrix representation
 */
///@{
template<> double SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::value() const;
template<> Eigen::SparseMatrix<double>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::row() const;
template<> Eigen::SparseMatrix<double>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >::InnerIterator::col() const;
///@}

/**
 * Single-precision float + Row-major sparse matrix representation
 */
 ///@{
template<> float SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::value() const;
template<> Eigen::SparseMatrix<float>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::row() const;
template<> Eigen::SparseMatrix<float>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >::InnerIterator::col() const;
///@}

/**
 * Double-precision float + Row-major sparse matrix representation
 */
///@{
template<> double SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::value() const;
template<> Eigen::SparseMatrix<double>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::row() const;
template<> Eigen::SparseMatrix<double>::Index SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >::InnerIterator::col() const;

#if !defined(SOFA_LINEARAGEBRA_SPARSEMATRIXTRANSPOSE_EIGENSPARSEMATRIX_CPP)
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float> >;
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double> >;

    extern template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<float, Eigen::RowMajor> >;
    extern template class SOFA_LINEARALGEBRA_API SparseMatrixStorageOrder<Eigen::SparseMatrix<double, Eigen::RowMajor> >;
#endif

} //namespace sofa::linearalgebra