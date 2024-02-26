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
#include <sofa/linearalgebra/SparseMatrixProduct.inl>
#include <sofa/testing/NumericTest.h>
#include <Sofa.LinearAlgebra.Testing/SparseMatrixTest.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <Eigen/Sparse>
#include <sofa/helper/random.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>


template <class TLHSMatrix, class TRHSMatrix, class TReal>
struct TestSparseMatrixProductTraits
{
    using LHSMatrix = TLHSMatrix;
    using RHSMatrix = TRHSMatrix;
    using Real = TReal;
};

/**
 * Test the class SparseMatrixProduct
 * The class is designed to use the templates defined in TestSparseMatrixProductTraits
 * The type of matrix can be any of the types supported by SparseMatrixProduct.
 */
template <class T>
struct TestSparseMatrixProduct : public sofa::testing::SparseMatrixTest<typename T::Real>
{
    using LHSMatrix = typename T::LHSMatrix;
    using RHSMatrix = typename T::RHSMatrix;
    using Real = typename T::Real;
    using Base = sofa::testing::SparseMatrixTest<typename T::Real>;
    using Base::generateRandomSparseMatrix;
    using Base::copyFromEigen;
    using Base::compareSparseMatrix;

    bool checkMatrix(typename LHSMatrix::Index nbRowsA, typename LHSMatrix::Index nbColsA, typename RHSMatrix::Index nbColsB, Real sparsity)
    {
        Eigen::SparseMatrix<Real, Eigen::RowMajor> eigen_a;
        Eigen::SparseMatrix<Real, Eigen::ColMajor> eigen_b;

        generateRandomSparseMatrix(eigen_a, nbRowsA, nbColsA, sparsity);
        generateRandomSparseMatrix(eigen_b, nbColsA, nbColsB, sparsity);

        LHSMatrix A;
        RHSMatrix B;
        copyFromEigen(A, eigen_a);
        copyFromEigen(B, eigen_b);

        EXPECT_GT(eigen_a.outerSize(), 0);
        EXPECT_GT(eigen_b.outerSize(), 0);

        Eigen::SparseMatrix<Real, Eigen::RowMajor> eigen_c = eigen_a * eigen_b;

        EXPECT_EQ(eigen_c.rows(), nbRowsA);
        EXPECT_EQ(eigen_c.cols(), nbColsB);
        EXPECT_GT(eigen_c.outerSize(), 0); //to make sure that there are non-zero values in the result matrix

        sofa::linearalgebra::SparseMatrixProduct<LHSMatrix, RHSMatrix, Eigen::SparseMatrix<Real> > product(&A, &B);
        product.computeProduct();

        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        //the second time computeProduct is called uses the faster algorithm
        product.computeProduct();
        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        // force re-computing the intersection
        product.computeProduct(true);
        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        //modify the values of A, but not its pattern
        for (int i = 0; i < eigen_a.nonZeros(); ++i)
        {
            eigen_a.valuePtr()[i] = static_cast<SReal>(sofa::helper::drand(1));
        }

        eigen_c = eigen_a * eigen_b; //result is updated using the regular matrix product
        copyFromEigen(A, eigen_a);

        product.m_lhs = &A;
        product.computeProduct(); //intersection is already computed: uses the faster algorithm
        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        return true;
    }
};

using CRSMatrixScalar = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;

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
TYPED_TEST_SUITE(TestSparseMatrixProduct, TestSparseMatrixProductImplementations);

TYPED_TEST(TestSparseMatrixProduct, squareMatrix )
{
    ASSERT_TRUE( this->checkMatrix( 5, 5, 5, 1. / 5. ) );
    ASSERT_TRUE( this->checkMatrix( 5, 5, 5, 3. / 5. ) );

    ASSERT_TRUE( this->checkMatrix( 100, 1000, 1000, 1. / 1000. ) );
    ASSERT_TRUE( this->checkMatrix( 1000, 1000, 1000, 20. / 1000. ) );

    ASSERT_TRUE( this->checkMatrix( 20, 20, 20, 1. ) );
}

TYPED_TEST(TestSparseMatrixProduct, rectangularMatrix )
{
    ASSERT_TRUE( this->checkMatrix( 5, 10, 7, 1. / 5. ) );
    ASSERT_TRUE( this->checkMatrix( 5, 10, 7, 3. / 5. ) );

    ASSERT_TRUE( this->checkMatrix( 10, 5, 7, 1. / 5. ) );
    ASSERT_TRUE( this->checkMatrix( 10, 5, 7, 3. / 5. ) );

    ASSERT_TRUE( this->checkMatrix( 1000, 3000, 2000, 1. / 1000. ) );
    ASSERT_TRUE( this->checkMatrix( 1000, 3000, 2000, 20. / 1000. ) );

    ASSERT_TRUE( this->checkMatrix( 20, 30, 10, 1. ) );
}
