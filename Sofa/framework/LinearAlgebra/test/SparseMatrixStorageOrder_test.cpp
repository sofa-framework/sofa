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
#include <sofa/linearalgebra/SparseMatrixStorageOrder[EigenSparseMatrix].h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <Sofa.LinearAlgebra.Testing/SparseMatrixTest.h>

template <class TMatrix, class TReal>
struct TestSparseMatrixTransposeTraits
{
    using Matrix = TMatrix;
    using Real = TReal;
};

template <class T>
struct TestSparseMatrixTranspose : public sofa::testing::SparseMatrixTest<typename T::Real>
{
    using Matrix = typename T::Matrix;
    using Real = typename T::Real;
    using Base = sofa::testing::SparseMatrixTest<typename T::Real>;
    using Base::generateRandomSparseMatrix;

    struct Triplet
    {
        typename Matrix::Index row;
        typename Matrix::Index col;
        Real value;
    };

    void checkMatrix(typename Matrix::Index nbRows, typename Matrix::Index nbCols, Real sparsity)
    {
        Matrix matrix;
        generateRandomSparseMatrix(matrix, nbRows, nbCols, sparsity);

        using Transpose = sofa::linearalgebra::SparseMatrixStorageOrder<Matrix>;
        const Transpose transposeMatrix(&matrix);

        const auto& outerStarts  = transposeMatrix.getOuterStarts();
        sofa::type::vector< std::pair< Triplet, Real> > issues;

        sofa::type::vector<Eigen::Triplet<Real> > triplets;
        for (std::size_t i = 0; i < outerStarts.size() - 1; ++i)
        {
            for(typename Transpose::InnerIterator it(transposeMatrix, i); it; ++it )
            {
                ASSERT_TRUE(it.row() < nbRows) << it.row() << " " << nbRows;
                ASSERT_TRUE(it.col() < nbCols) << it.col() << " " << nbCols;
                const Real initialValue = matrix.coeff(it.row(), it.col());

                triplets.emplace_back(it.row(), it.col(), it.value());

                if (!this->isSmall(initialValue - it.value()))
                {
                    const Triplet t{it.row(), it.col(), static_cast<Real>(it.value())};
                    std::pair<Triplet, Real> p(t, initialValue);
                    issues.push_back(p);
                }
            }
        }

        if (!issues.empty())
        {
            const std::stringstream ss;
            ADD_FAILURE() << "Found " << issues.size() << " differences in the transposed matrix compared to the initial matrix\n" << ss.str();
        }

    }
};

using CRSMatrixScalar = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;

using TestSparseMatrixTransposeImplementations = ::testing::Types<
    TestSparseMatrixTransposeTraits<Eigen::SparseMatrix<float>, float>,
    TestSparseMatrixTransposeTraits<Eigen::SparseMatrix<double>, double>,
    TestSparseMatrixTransposeTraits<Eigen::SparseMatrix<float, Eigen::RowMajor>, float>,
    TestSparseMatrixTransposeTraits<Eigen::SparseMatrix<double, Eigen::RowMajor>, double>
    // TestSparseMatrixTransposeTraits<CRSMatrixScalar, SReal, 1000, std::ratio<1, 1000> >
>;
TYPED_TEST_SUITE(TestSparseMatrixTranspose, TestSparseMatrixTransposeImplementations);

TYPED_TEST(TestSparseMatrixTranspose, squareMatrix )
{
    this->checkMatrix( 5, 5, 1. / 5. );
    this->checkMatrix( 5, 5, 3. / 5. );

    this->checkMatrix( 1000, 1000, 1. / 1000. );
    this->checkMatrix( 1000, 1000, 20. / 1000. );

    this->checkMatrix( 100, 100, 1. );
}

TYPED_TEST(TestSparseMatrixTranspose, rectangularMatrix )
{
    this->checkMatrix( 5, 10, 1. / 5. );
    this->checkMatrix( 5, 10, 3. / 5. );

    this->checkMatrix( 10, 5, 1. / 5. );
    this->checkMatrix( 10, 5, 3. / 5. );

    this->checkMatrix( 1000, 3000, 1. / 1000. );
    this->checkMatrix( 1000, 3000, 20. / 1000. );

    this->checkMatrix( 100, 300, 1. );
}