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
#include <gtest/gtest.h>
#include <sofa/testing/NumericTest.h>
#include <Sofa.LinearAlgebra.Testing/SparseMatrixTest.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <Eigen/Sparse>
#include <sofa/helper/random.h>

namespace sofa::linearalgebra::testing
{

template<class T>
struct SparseMatrixProductInit
{
    static void init(T& product)
    {
        SOFA_UNUSED(product);
    };

    static void cleanup(T& product)
    {
        SOFA_UNUSED(product);
    }
};

/**
 * Test the class SparseMatrixProduct
 * The class is designed to use the templates defined in TestSparseMatrixProductTraits
 * The type of matrix can be any of the types supported by SparseMatrixProduct.
 */
template <class T>
struct TestSparseMatrixProduct : public sofa::testing::SparseMatrixTest<typename T::LhsScalar>
{
    using LHSMatrix = typename T::LhsCleaned;
    using RHSMatrix = typename T::RhsCleaned;
    using Real = typename T::LhsScalar;
    using Base = sofa::testing::SparseMatrixTest<Real>;
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

        T product(&A, &B);
        SparseMatrixProductInit<T>::init(product);
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

        SparseMatrixProductInit<T>::cleanup(product);

        return true;
    }
};

TYPED_TEST_SUITE_P(TestSparseMatrixProduct);

TYPED_TEST_P(TestSparseMatrixProduct, squareMatrix )
{
    EXPECT_TRUE( this->checkMatrix( 5, 5, 5, 1. / 5. ) );
    EXPECT_TRUE( this->checkMatrix( 5, 5, 5, 3. / 5. ) );

    EXPECT_TRUE( this->checkMatrix( 100, 1000, 1000, 1. / 1000. ) );
    EXPECT_TRUE( this->checkMatrix( 1000, 1000, 1000, 20. / 1000. ) );
    // EXPECT_TRUE( this->checkMatrix( 1000, 1000, 1000, 150. / 1000. ) );

    EXPECT_TRUE( this->checkMatrix( 20, 20, 20, 1. ) );
}

TYPED_TEST_P(TestSparseMatrixProduct, rectangularMatrix )
{
    EXPECT_TRUE( this->checkMatrix( 5, 10, 7, 1. / 5. ) );
    EXPECT_TRUE( this->checkMatrix( 5, 10, 7, 3. / 5. ) );

    EXPECT_TRUE( this->checkMatrix( 10, 5, 7, 1. / 5. ) );
    EXPECT_TRUE( this->checkMatrix( 10, 5, 7, 3. / 5. ) );

    EXPECT_TRUE( this->checkMatrix( 1000, 3000, 2000, 1. / 1000. ) );
    EXPECT_TRUE( this->checkMatrix( 1000, 3000, 2000, 20. / 1000. ) );

    EXPECT_TRUE( this->checkMatrix( 20, 30, 10, 1. ) );
}

REGISTER_TYPED_TEST_SUITE_P(TestSparseMatrixProduct,
                            squareMatrix,rectangularMatrix);

}
