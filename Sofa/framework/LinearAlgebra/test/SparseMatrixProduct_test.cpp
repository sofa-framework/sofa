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
#include <Sofa.LinearAlgebra.Testing/SparseMatrixProduct_test.h>
#include <sofa/linearalgebra/SparseMatrixProduct.inl>

namespace sofa
{

using namespace sofa::linearalgebra::testing;

using TestSparseMatrixProductImplementations = ::testing::Types<
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<float>, Eigen::SparseMatrix<float>, float>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>, double>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::SparseMatrix<float, Eigen::RowMajor>, float>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::SparseMatrix<double, Eigen::RowMajor>, double>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<float, Eigen::ColMajor>, Eigen::SparseMatrix<float, Eigen::RowMajor>, float>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<double, Eigen::ColMajor>, Eigen::SparseMatrix<double, Eigen::RowMajor>, double>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::SparseMatrix<float, Eigen::ColMajor>, float>,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::SparseMatrix<double, Eigen::ColMajor>, double>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    TestSparseMatrixProduct,
    TestSparseMatrixProduct,
    TestSparseMatrixProductImplementations
);

}
