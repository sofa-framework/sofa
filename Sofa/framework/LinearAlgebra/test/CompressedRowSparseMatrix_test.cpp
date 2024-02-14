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
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

#include <Eigen/Sparse>

#include <sofa/testing/NumericTest.h>

#include <sofa/helper/RandomGenerator.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>


TEST(matrix_bloc_traits, subBlock)
{
    sofa::type::Mat<6, 6, SReal> mat6x6;

    sofa::testing::LinearCongruentialRandomGenerator lcg(46515387);
    for (sofa::Size i = 0; i < mat6x6.nbLines; i++)
    {
        for (sofa::Size j = 0; j < mat6x6.nbCols; j++)
        {
            mat6x6(i, j) = lcg.generateInRange(0., 1.);
        }
    }

    for (const auto& [r, c] : sofa::type::vector<std::pair<sofa::Index, sofa::Index>>{{0, 0}, {0, 3}, {3, 0}, {3, 3}, {1, 2}})
    {
        sofa::type::Mat<3, 3, SReal> mat3x3;
        sofa::linearalgebra::matrix_bloc_traits<sofa::type::Mat<6, 6, SReal>, sofa::Index>::subBlock(mat6x6, r, c, mat3x3);

        for (sofa::Size i = 0; i < mat3x3.nbLines; i++)
        {
            for (sofa::Size j = 0; j < mat3x3.nbCols; j++)
            {
                EXPECT_EQ(mat6x6(i + r, j + c), mat3x3(i, j));
            }
        }
    }

    for (const auto& [r, c] : sofa::type::vector<std::pair<sofa::Index, sofa::Index>>{{0, 0}, {0, 3}, {3, 0}, {3, 3}, {1, 2}})
    {
        SReal real;
        sofa::linearalgebra::matrix_bloc_traits<sofa::type::Mat<6, 6, SReal>, sofa::Index>::subBlock(mat6x6, r, c, real);
        EXPECT_EQ(real, mat6x6(r, c));
    }

}

template<typename TBlock>
void generateMatrix(sofa::linearalgebra::CompressedRowSparseMatrix<TBlock>& matrix,
    sofa::SignedIndex nbRows, sofa::SignedIndex nbCols,
    typename sofa::linearalgebra::CompressedRowSparseMatrix<TBlock>::Real sparsity,
    long seed)
{
    using Real = typename sofa::linearalgebra::CompressedRowSparseMatrix<TBlock>::Real;
    const auto nbNonZero = static_cast<sofa::SignedIndex>(sparsity * static_cast<Real>(nbRows*nbCols));

    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed(seed);

    matrix.resize(nbRows, nbCols);

    for (sofa::SignedIndex i = 0; i < nbNonZero; ++i)
    {
        const auto value = static_cast<Real>(sofa::helper::drand(1));
        const auto row = randomGenerator.random<sofa::Index>(0, nbRows);
        const auto col = randomGenerator.random<sofa::Index>(0, nbCols);
        matrix.add(row, col, value);
    }
    matrix.compress();
}

/**
 * Two matrices A and B are generated randomly as CompressedRowSparseMatrix.
 * The test checks the consistency of the results of A^T * B, computed using 3 methods:
 * 1) CompressedRowSparseMatrix::mulTranspose
 * 2) Both A and B are converted to Eigen::SparseMatrix, then A.transpose() * B
 * 3) Both A and B are mapped to Eigen::Map, then A.transpose() * B
 * To test the 3 methods give the same result, the three results are converted to a list of triplets. The three lists
 * are then compared.
 */
TEST(CompressedRowSparseMatrix, transposeProduct)
{
    constexpr int ROW_A = 1024;
    constexpr int ROW_B = 1024;
    constexpr int COL_A = 512;
    constexpr int COL_B = 256;

    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> A, B, C;
    generateMatrix(A, ROW_A, COL_A, 0.01, 12);
    generateMatrix(B, ROW_B, COL_B, 0.01, 13);

    // The following operation is required to be able to map a CompressedRowSparseMatrix to a Eigen::Map
    A.fullRows();
    B.fullRows();

    Eigen::SparseMatrix<SReal, Eigen::RowMajor> AEigen, BEigen;

    //inefficient conversion from CompressedRowSparseMatrix to a Eigen::SparseMatrix
    const auto toEigen = [](const sofa::linearalgebra::CompressedRowSparseMatrix<SReal>& crs, Eigen::SparseMatrix<SReal, Eigen::RowMajor>& eigenMatrix)
    {
        eigenMatrix.resize(crs.rows(), crs.cols());
        std::vector<Eigen::Triplet<SReal> > triplets;
        triplets.reserve(crs.colsValue.size());
        for (unsigned int it_rows_k=0; it_rows_k < crs.rowIndex.size() ; it_rows_k ++)
        {
            const auto row = crs.rowIndex[it_rows_k];
            typename sofa::linearalgebra::CompressedRowSparseMatrix<SReal>::Range rowRange( crs.rowBegin[it_rows_k], crs.rowBegin[it_rows_k+1] );
            for(auto xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
            {
                const auto col = crs.colsIndex[xj];
                const auto k = crs.colsValue[xj];
                triplets.emplace_back(row, col, k);
            }
        }
        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    };
    toEigen(A, AEigen);
    toEigen(B, BEigen);


    // Compute C = A^T * B using CompressedRowSparseMatrix
    A.mulTranspose(C, B); // C = A^T * B

    EXPECT_EQ(C.rows(), COL_A);
    EXPECT_EQ(C.cols(), COL_B);

    // Compute C = A^T * B using Eigen operations
    const Eigen::SparseMatrix<SReal, Eigen::RowMajor> CEigen = AEigen.transpose() * BEigen;

    EXPECT_EQ(CEigen.rows(), COL_A);
    EXPECT_EQ(CEigen.cols(), COL_B);


    using EigenMap = Eigen::Map<const Eigen::SparseMatrix<SReal, Eigen::RowMajor> >;
    EigenMap AMap
        (A.rows(), A.cols(), A.getColsValue().size(),
        (EigenMap::StorageIndex*)A.rowBegin.data(), (EigenMap::StorageIndex*)A.colsIndex.data(), A.colsValue.data());
    EigenMap BMap
        (B.rows(), B.cols(), B.getColsValue().size(),
        (EigenMap::StorageIndex*)B.rowBegin.data(), (EigenMap::StorageIndex*)B.colsIndex.data(), B.colsValue.data());

    // Compute C = A^T * B using Eigen operations on Eigen::Map
    const Eigen::SparseMatrix<SReal, Eigen::RowMajor> CEigenMap = (AMap.transpose() * BMap).pruned();

    EXPECT_EQ(CEigenMap.rows(), COL_A);
    EXPECT_EQ(CEigenMap.cols(), COL_B);


    //Conversion of the three results to a list of triplets
    std::vector<std::tuple<int, int, SReal> > triplets_CRS, triplets_Eigen, triplets_EigenMap;

    for (int k = 0; k < CEigen.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<SReal, Eigen::RowMajor>::InnerIterator it(CEigen, k); it; ++it)
        {
            triplets_Eigen.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (int k = 0; k < CEigenMap.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<SReal, Eigen::RowMajor>::InnerIterator it(CEigenMap, k); it; ++it)
        {
            triplets_EigenMap.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (unsigned int it_rows_k=0; it_rows_k < C.rowIndex.size() ; it_rows_k ++)
    {
        const auto row = C.rowIndex[it_rows_k];
        decltype(C)::Range rowRange( C.rowBegin[it_rows_k], C.rowBegin[it_rows_k+1] );
        for(auto xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
        {
            const auto col = C.colsIndex[xj];
            const auto k = C.colsValue[xj];
            triplets_CRS.emplace_back(row, col, k);
        }
    }

    //Comparison of the 3 lists of triplets

    EXPECT_EQ(triplets_Eigen.size(), triplets_EigenMap.size());
    EXPECT_EQ(triplets_CRS.size(), triplets_EigenMap.size());

    if (triplets_Eigen.size() == triplets_EigenMap.size())
    {
        for (unsigned int i = 0; i < triplets_Eigen.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(triplets_Eigen[i]), std::get<0>(triplets_EigenMap[i]));
            EXPECT_EQ(std::get<1>(triplets_Eigen[i]), std::get<1>(triplets_EigenMap[i]));
            EXPECT_NEAR(std::get<2>(triplets_Eigen[i]), std::get<2>(triplets_EigenMap[i]), 1e-10);
        }
    }

    if (triplets_CRS.size() == triplets_EigenMap.size())
    {
        for (unsigned int i = 0; i < triplets_CRS.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(triplets_CRS[i]), std::get<0>(triplets_EigenMap[i]));
            EXPECT_EQ(std::get<1>(triplets_CRS[i]), std::get<1>(triplets_EigenMap[i]));
            EXPECT_NEAR(std::get<2>(triplets_CRS[i]), std::get<2>(triplets_EigenMap[i]), 1e-10);
        }
    }
}

TEST(CompressedRowSparseMatrix, fullRowsNoEntries)
{
    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> A;
    A.resize(1321, 3556);
    EXPECT_TRUE(A.getRowIndex().empty());
    EXPECT_NO_THROW(A.fullRows());
    EXPECT_EQ(A.getRowIndex().size(), 1321);

    //make sure that we can iterate, but the content is empty

    std::vector<std::tuple<int, int, SReal> > triplets_CRS;
    for (unsigned int it_rows_k=0; it_rows_k < A.rowIndex.size() ; it_rows_k ++)
    {
        const auto row = A.rowIndex[it_rows_k];
        decltype(A)::Range rowRange( A.rowBegin[it_rows_k], A.rowBegin[it_rows_k+1] );
        for(auto xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )
        {
            const auto col = A.colsIndex[xj];
            const auto k = A.colsValue[xj];
            triplets_CRS.emplace_back(row, col, k);
        }
    }
    EXPECT_TRUE(triplets_CRS.empty());
}

TEST(CompressedRowSparseMatrix, fullRowsWithEntries)
{
    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> A;
    generateMatrix(A, 1321, 3556, 0.0003, 12);
    EXPECT_FALSE(A.getRowIndex().empty());
    EXPECT_NO_THROW(A.fullRows());
    EXPECT_EQ(A.getRowIndex().size(), 1321);
}


TEST(CompressedRowSparseMatrix, copyNonZeros)
{
    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> A;
    generateMatrix(A, 1321, 3556, 0.0003, 12);

    const auto numberNonZeroValues1 = A.colsValue.size();

    A.add(23, 569, 0);
    A.add(874, 326, 0);
    A.add(769, 1789, 0);
    A.compress();

    const auto numberNonZeroValues2 = A.colsValue.size();
    EXPECT_GT(numberNonZeroValues2, numberNonZeroValues1);

    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> B;

    B.copyNonZeros(A);
    const auto numberNonZeroValues3 = B.colsValue.size();

    EXPECT_EQ(B.rowBSize(), A.rowBSize());
    EXPECT_EQ(B.colBSize(), A.colBSize());

    EXPECT_EQ(B.rowSize(), A.rowSize());
    EXPECT_EQ(B.colSize(), A.colSize());

    EXPECT_EQ(numberNonZeroValues1, numberNonZeroValues3);
}

TEST(CompressedRowSparseMatrix, copyNonZerosFrom3x3Blocks)
{
    sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Mat<3, 3, SReal>> A;
    generateMatrix(A, 1321, 3556, 0.0003, 12);

    const auto numberNonZeroValues1 = A.colsValue.size();

    A.add(23, 569, 0);
    A.add(874, 326, 0);
    A.add(769, 1789, 0);
    A.compress();

    const auto numberNonZeroValues2 = A.colsValue.size();
    EXPECT_GT(numberNonZeroValues2, numberNonZeroValues1);

    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> B;

    B.copyNonZeros(A);

    for (unsigned int r = 0; r < A.rowSize(); ++r)
    {
        for (unsigned int c = 0; c < A.colSize(); ++c)
        {
            EXPECT_NEAR(A(r, c), B(r, c), 1e-12_sreal) << "r = " << r << ", c = " << c;
        }
    }
}

TEST(CompressedRowSparseMatrix, copyNonZerosFrom1x3Blocks)
{
    sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Vec<3, SReal>> A;
    generateMatrix(A, 1321, 3556, 0.0003, 12);

    const auto numberNonZeroValues1 = A.colsValue.size();

    A.add(23, 569, 0);
    A.add(874, 326, 0);
    A.add(769, 1789, 0);
    A.compress();

    const auto numberNonZeroValues2 = A.colsValue.size();
    EXPECT_GT(numberNonZeroValues2, numberNonZeroValues1);

    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> B;

    B.copyNonZeros(A);

    for (unsigned int r = 0; r < A.rowSize(); ++r)
    {
        for (unsigned int c = 0; c < A.colSize(); ++c)
        {
            EXPECT_NEAR(A(r, c), B(r, c), 1e-12_sreal) << "r = " << r << ", c = " << c;
        }
    }
}

TEST(CompressedRowSparseMatrix, emptyMatrixGetRowRange)
{
    EXPECT_EQ(sofa::linearalgebra::CompressedRowSparseMatrixMechanical<SReal>::s_invalidIndex, std::numeric_limits<sofa::SignedIndex>::lowest());

    const sofa::linearalgebra::CompressedRowSparseMatrixMechanical<SReal> A;

    const auto range = A.getRowRange(0);
    EXPECT_EQ(range.first, sofa::linearalgebra::CompressedRowSparseMatrixMechanical<SReal>::s_invalidIndex);
    EXPECT_EQ(range.second, sofa::linearalgebra::CompressedRowSparseMatrixMechanical<SReal>::s_invalidIndex);
}

TEST(CompressedRowSparseMatrixConstraint, emptyMatrixGetRowRange)
{
    EXPECT_EQ(sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex, std::numeric_limits<sofa::SignedIndex>::lowest());

    const sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal> A;

    const auto begin = A.begin();
    EXPECT_EQ(begin.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);

    const auto end = A.end();
    EXPECT_EQ(end.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);

    EXPECT_EQ(begin, end);

    const auto checkIterator = [](const auto& iterator)
    {
        const auto itBegin = iterator.begin();
        const auto itEnd = iterator.end();

        EXPECT_EQ(itBegin.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);
        EXPECT_EQ(itEnd.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);
        EXPECT_EQ(itBegin, itEnd);
    };

    checkIterator(begin);
    checkIterator(end);
}
