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
#include <ratio>
#include <sofa/linearalgebra/SparseMatrixProduct[CompressedRowSparseMatrix].h>
#include <sofa/linearalgebra/SparseMatrixProduct[EigenSparseMatrix].h>
#include <sofa/testing/NumericTest.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <Eigen/Sparse>
#include <sofa/helper/random.h>

template <class TMatrix, class TReal, sofa::linearalgebra::BaseMatrix::Index TSquareMatrixSize, class TSparsity>
struct TestSparseMatrixProductTraits
{
    using Matrix = TMatrix;
    using Real = TReal;
    static constexpr sofa::linearalgebra::BaseMatrix::Index SquareMatrixSize = TSquareMatrixSize;
    static_assert(TSparsity::num <= TSparsity::den && TSparsity::num > 0 && TSparsity::den > 0, "Must be a ratio between 0 and 1");
    static constexpr TReal Sparsity = static_cast<TReal>(TSparsity::num) / static_cast<TReal>(TSparsity::den);
};

/**
 * Test the class SparseMatrixProduct
 * The class is designed to use the templates defined in TestSparseMatrixProductTraits
 * The type of matrix can be any of the types supported by SparseMatrixProduct.
 */
template <class T>
struct TestSparseMatrixProduct : public sofa::testing::NumericTest<typename T::Real>
{
    using Matrix = typename T::Matrix;
    using Real = typename T::Real;
    static constexpr sofa::linearalgebra::BaseMatrix::Index SquareMatrixSize = T::SquareMatrixSize;
    static constexpr Real Sparsity = T::Sparsity;
    static_assert(Sparsity > 0 && Sparsity <= 1, "Must be a ratio between 0 and 1");

    void generateRandomSparseMatrix(Eigen::SparseMatrix<Real>& eigenMatrix) const
    {
        eigenMatrix.resize(SquareMatrixSize, SquareMatrixSize);
        sofa::type::vector<Eigen::Triplet<SReal> > triplets;

        static constexpr sofa::linearalgebra::BaseMatrix::Index nbNonZero = Sparsity * static_cast<Real>(SquareMatrixSize*SquareMatrixSize);
        static_assert(nbNonZero > 0, "Number of non zero is not > 0");

        for (sofa::linearalgebra::BaseMatrix::Index i = 0; i < nbNonZero; ++i)
        {
            const auto value = static_cast<SReal>(sofa::helper::drand(1));
            const auto row = static_cast<sofa::linearalgebra::BaseMatrix::Index>(sofa::helper::drandpos(SquareMatrixSize));
            const auto col = static_cast<sofa::linearalgebra::BaseMatrix::Index>(sofa::helper::drandpos(SquareMatrixSize));
            triplets.emplace_back(row, col, value);
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    void copyFromEigen(Matrix& dst, const Eigen::SparseMatrix<Real>& src)
    {
        if constexpr (std::is_same_v<Matrix, Eigen::SparseMatrix<Real> >)
        {
            dst = src;
        }
        else
        {
            dst.resize(src.rows(), src.cols());
            for (typename Eigen::SparseMatrix<Real>::Index k = 0; k < src.outerSize(); ++k)
            {
                for (typename Eigen::SparseMatrix<Real>::InnerIterator it(src, k); it; ++it)
                {
                    dst.add(it.row(), it.col(), it.value());
                }
            }
        }
    }

    bool compareSparseMatrix(const Eigen::SparseMatrix<Real>& A, const Matrix& B)
    {
        if constexpr (std::is_same_v<Matrix, Eigen::SparseMatrix<Real> >)
        {
            return compareEigenSparseMatrix(A, B);
        }
        else
        {
            Eigen::SparseMatrix<Real> copy;
            copy.resize(B.rows(), B.cols());

            sofa::type::vector<Eigen::Triplet<Real> > triplets;
            for (unsigned int it_rows_k=0; it_rows_k < B.rowIndex.size() ; it_rows_k ++)
            {
                const auto row = B.rowIndex[it_rows_k];
                typename Matrix::Range rowRange( B.rowBegin[it_rows_k], B.rowBegin[it_rows_k+1] );
                for(auto xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
                {
                    const auto col = B.colsIndex[xj];
                    const auto k = B.colsValue[xj];
                    triplets.emplace_back(row * Matrix::NL, col * Matrix::NC, k);
                }
            }

            copy.setFromTriplets(triplets.begin(), triplets.end());

            return compareEigenSparseMatrix(A, copy);
        }
    }

    bool compareEigenSparseMatrix(const Eigen::SparseMatrix<Real>& A, const Eigen::SparseMatrix<Real>& B)
    {
        if (A.outerSize() != B.outerSize())
            return false;

        for (int k=0; k < A.outerSize(); ++k)
        {
            sofa::type::vector<Eigen::Triplet<Real> > triplets_a, triplets_b;
            for (typename Eigen::SparseMatrix<Real>::InnerIterator it(A, k); it; ++it)
            {
                triplets_a.emplace_back(it.row(), it.col(), it.value());
            }
            for (typename Eigen::SparseMatrix<Real>::InnerIterator it(B, k); it; ++it)
            {
                triplets_b.emplace_back(it.row(), it.col(), it.value());
            }

            if (triplets_a.size() != triplets_b.size())
                return false;

            for (size_t i = 0 ; i < triplets_a.size(); ++i)
            {
                const auto& a = triplets_a[i];
                const auto& b = triplets_b[i];

                if (a.row() != b.row() && a.col() != b.col() && a.value() != b.value())
                {
                    return false;
                }
            }

        }
        return true;
    }

    bool checkSquareMatrix()
    {
        Eigen::SparseMatrix<Real> eigen_a, eigen_b;
        generateRandomSparseMatrix(eigen_a);
        generateRandomSparseMatrix(eigen_b);

        Matrix A, B;
        copyFromEigen(A, eigen_a);
        copyFromEigen(B, eigen_b);

        EXPECT_GT(eigen_a.outerSize(), 0);
        EXPECT_GT(eigen_b.outerSize(), 0);

        const Eigen::SparseMatrix<Real> eigen_c = eigen_a * eigen_b;
        EXPECT_EQ(eigen_c.rows(), SquareMatrixSize);
        EXPECT_EQ(eigen_c.cols(), SquareMatrixSize);
        EXPECT_GT(eigen_c.outerSize(), 0); //to make sure that there are non-zero values in the result matrix

        sofa::linearalgebra::SparseMatrixProduct<Matrix> product(&A, &B);
        product.computeProduct();

        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        product.computeProduct();
        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        product.computeProduct(true);
        EXPECT_TRUE(compareSparseMatrix(eigen_c, product.getProductResult()));

        return true;
    }
};

using CRSMatrixScalar = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;

using TestSparseMatrixProductImplementations = ::testing::Types<
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<SReal>, SReal, 10, std::ratio<1, 10> >,
    TestSparseMatrixProductTraits<Eigen::SparseMatrix<SReal>, SReal, 1000, std::ratio<1, 1000> >,
    TestSparseMatrixProductTraits<CRSMatrixScalar, SReal, 1000, std::ratio<1, 1000> >
>;
TYPED_TEST_SUITE(TestSparseMatrixProduct, TestSparseMatrixProductImplementations);

TYPED_TEST(TestSparseMatrixProduct, squareMatrix ) { ASSERT_TRUE( this->checkSquareMatrix() ); }