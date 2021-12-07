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
#include <Sofa.LinearAlgebra.Testing/BaseMatrix_test.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>

namespace sofa
{
using namespace sofa::linearalgebra::testing;

/// definition of a EigenSparseMatrix type used in tests
template<sofa::Index TNbRows, sofa::Index TNbCols, class TReal >
using TestEigenMatrix = linearalgebra::EigenSparseMatrix<
    defaulttype::StdVectorTypes< sofa::type::Vec<TNbCols, TReal>, sofa::type::Vec<TNbCols, TReal> >,
    defaulttype::StdVectorTypes< sofa::type::Vec<TNbRows, TReal>, sofa::type::Vec<TNbRows, TReal> >
>;

/// list of types tested in TestBaseMatrix
template<sofa::Index TNbRows, sofa::Index TNbCols, class TReal >
using TestBaseMatrixTypes = ::testing::Types<
    TestBaseMatrixTraits< linearalgebra::FullMatrix<TReal>, TNbRows, TNbCols, TReal>,
    TestBaseMatrixTraits< linearalgebra::CompressedRowSparseMatrix<TReal>, TNbRows, TNbCols, TReal >,
    TestBaseMatrixTraits< linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, TReal> >, TNbRows, TNbCols, TReal >,
    TestBaseMatrixTraits< linearalgebra::SparseMatrix<TReal>, TNbRows, TNbCols, TReal >,
    TestBaseMatrixTraits< TestEigenMatrix<TNbRows, TNbCols, TReal>, TNbRows, TNbCols, TReal >
>;

/// List of 9x9 matrix types that will be instantiated
template<class TReal>
using TestBaseMatrix9x9Types = TestBaseMatrixTypes<9,9,TReal>;

/// Test instantiation for 9x9 matrices and float scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaBaseLinearSolver_test_9x9float,
    TestBaseMatrix,
    TestBaseMatrix9x9Types<float>
);
/// Test instantiation for 9x9 matrices and double scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaBaseLinearSolver_test_9x9double,
    TestBaseMatrix,
    TestBaseMatrix9x9Types<double>
);

/// List of 9x14 matrix types that will be instantiated
template<class TReal>
using TestBaseMatrix9x14Types = TestBaseMatrixTypes<9,14,TReal>;

/// Test instantiation for 9x14 matrices and float scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaBaseLinearSolver_test_9x14float,
    TestBaseMatrix,
    TestBaseMatrix9x14Types<float>
);
/// Test instantiation for 9x14 matrices and double scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaBaseLinearSolver_test_9x14double,
    TestBaseMatrix,
    TestBaseMatrix9x14Types<double>
);

} //namespace sofa
