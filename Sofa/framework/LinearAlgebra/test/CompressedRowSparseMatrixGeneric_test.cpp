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
#include <sofa/linearalgebra/CompressedRowSparseMatrixGeneric.h>
#include <sofa/type/Mat.h>
#include <gtest/gtest.h>

using CRS = sofa::linearalgebra::CompressedRowSparseMatrixGeneric<double>;
using Mat3 = sofa::type::Mat<3, 3, double>;
using CRSMat3 = sofa::linearalgebra::CompressedRowSparseMatrixGeneric<Mat3>;

constexpr double kTol = 1e-10;

// ==================== Construction & Sizing ====================

TEST(CompressedRowSparseMatrixGeneric, DefaultConstruction)
{
    CRS m;
    EXPECT_EQ(m.rowBSize(), 0);
    EXPECT_EQ(m.colBSize(), 0);
    EXPECT_TRUE(m.getRowIndex().empty());
    EXPECT_TRUE(m.getRowBegin().empty());
    EXPECT_TRUE(m.getColsIndex().empty());
    EXPECT_TRUE(m.getColsValue().empty());
}

TEST(CompressedRowSparseMatrixGeneric, ConstructWithDimensions)
{
    CRS m(3, 4);
    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 4);
}

TEST(CompressedRowSparseMatrixGeneric, ResizeBlock)
{
    CRS m;
    m.resizeBlock(3, 4);
    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 4);

    m.resizeBlock(5, 6);
    EXPECT_EQ(m.rowBSize(), 5);
    EXPECT_EQ(m.colBSize(), 6);
}

TEST(CompressedRowSparseMatrixGeneric, ResizeBlockSameSize)
{
    CRS m(3, 4);
    m.setBlock(0, 0, 1.0);
    m.setBlock(1, 2, 5.0);
    m.compress();

    const auto numValues = m.getColsValue().size();
    m.resizeBlock(3, 4);

    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 4);
    EXPECT_EQ(m.getColsValue().size(), numValues);
    for (const auto& v : m.getColsValue())
    {
        EXPECT_NEAR(v, 0.0, kTol);
    }
}

TEST(CompressedRowSparseMatrixGeneric, ResizeBlockNonSquare)
{
    // Regression test for the nBlockRow==nbBRow && nBlockCol==nbBCol fix
    CRS m(3, 4);
    m.setBlock(0, 0, 1.0);
    m.setBlock(1, 3, 7.0);
    m.compress();

    // Same rows, different cols: must fully reset, not just zero
    m.resizeBlock(3, 6);
    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 6);
    EXPECT_TRUE(m.getRowIndex().empty());
    EXPECT_TRUE(m.getColsValue().empty());

    // Same cols, different rows: must fully reset
    CRS m2(3, 4);
    m2.setBlock(0, 0, 2.0);
    m2.compress();
    m2.resizeBlock(5, 4);
    EXPECT_EQ(m2.rowBSize(), 5);
    EXPECT_EQ(m2.colBSize(), 4);
    EXPECT_TRUE(m2.getRowIndex().empty());
    EXPECT_TRUE(m2.getColsValue().empty());
}

// ==================== Block Insertion & Retrieval ====================

TEST(CompressedRowSparseMatrixGeneric, SetBlockAndBlock)
{
    CRS m(4, 4);
    m.setBlock(1, 2, 3.14);
    EXPECT_NEAR(m.block(1, 2), 3.14, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, AddBlock)
{
    CRS m(4, 4);
    m.setBlock(0, 0, 3.0);
    m.compress();
    m.addBlock(0, 0, 2.0);
    EXPECT_NEAR(m.block(0, 0), 5.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, AddBlockMultipleToSamePosition)
{
    CRS m(4, 4);
    m.addBlock(1, 0, 2.0);
    m.addBlock(1, 1, 3.0);
    m.addBlock(1, 0, 1.0); // separate btemp entry since last was (1,1)
    m.compress();
    EXPECT_NEAR(m.block(1, 0), 3.0, kTol); // 2.0 + 1.0
    EXPECT_NEAR(m.block(1, 1), 3.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, WblockCreate)
{
    CRS m(4, 4);
    double* ptr = m.wblock(2, 3, true);
    ASSERT_NE(ptr, nullptr);
    *ptr = 42.0;
    EXPECT_NEAR(m.block(2, 3), 42.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, WblockNoCreate)
{
    CRS m(4, 4);
    double* ptr = m.wblock(2, 3, false);
    EXPECT_EQ(ptr, nullptr);
}

TEST(CompressedRowSparseMatrixGeneric, GetBlock)
{
    CRS m(4, 4);
    m.setBlock(1, 2, 7.5);
    const double& ref = m.getBlock(1, 2);
    EXPECT_NEAR(ref, 7.5, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, InsertionOrder)
{
    CRS m(6, 6);
    m.setBlock(4, 1, 10.0);
    m.setBlock(0, 3, 20.0);
    m.setBlock(2, 0, 30.0);
    m.setBlock(4, 5, 40.0);
    m.setBlock(0, 0, 50.0);

    // AutoCompress on read verifies all blocks
    EXPECT_NEAR(m.block(0, 0), 50.0, kTol);
    EXPECT_NEAR(m.block(0, 3), 20.0, kTol);
    EXPECT_NEAR(m.block(2, 0), 30.0, kTol);
    EXPECT_NEAR(m.block(4, 1), 10.0, kTol);
    EXPECT_NEAR(m.block(4, 5), 40.0, kTol);
    EXPECT_NEAR(m.block(1, 1), 0.0, kTol);
}

// ==================== Compression ====================

TEST(CompressedRowSparseMatrixGeneric, CompressAfterInsert)
{
    CRS m(4, 5);
    m.setBlock(0, 1, 1.0);
    m.setBlock(0, 3, 2.0);
    m.setBlock(2, 0, 3.0);
    m.setBlock(2, 4, 4.0);
    m.compress();

    const auto& ri = m.getRowIndex();
    const auto& rb = m.getRowBegin();
    const auto& ci = m.getColsIndex();
    const auto& cv = m.getColsValue();

    ASSERT_EQ(ri.size(), 2u);
    EXPECT_EQ(ri[0], 0);
    EXPECT_EQ(ri[1], 2);

    ASSERT_EQ(rb.size(), 3u);
    EXPECT_EQ(rb[0], 0);
    EXPECT_EQ(rb[1], 2);
    EXPECT_EQ(rb[2], 4);

    ASSERT_EQ(ci.size(), 4u);
    EXPECT_EQ(ci[0], 1);
    EXPECT_EQ(ci[1], 3);
    EXPECT_EQ(ci[2], 0);
    EXPECT_EQ(ci[3], 4);

    ASSERT_EQ(cv.size(), 4u);
    EXPECT_NEAR(cv[0], 1.0, kTol);
    EXPECT_NEAR(cv[1], 2.0, kTol);
    EXPECT_NEAR(cv[2], 3.0, kTol);
    EXPECT_NEAR(cv[3], 4.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, CompressRemovesZeros)
{
    CRS m(4, 4);
    m.setBlock(0, 0, 5.0);
    m.setBlock(0, 1, 3.0);
    m.setBlock(1, 0, 1.0);
    m.addBlock(0, 0, -5.0); // separate btemp entry, net (0,0) = 0

    m.compress();

    // The zero at (0,0) should not be in colsValue; only (0,1)=3 and (1,0)=1 remain
    EXPECT_EQ(m.getColsValue().size(), 2u);
    EXPECT_NEAR(m.block(0, 0), 0.0, kTol);
    EXPECT_NEAR(m.block(0, 1), 3.0, kTol);
    EXPECT_NEAR(m.block(1, 0), 1.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, CompressMergesBtemp)
{
    CRS m(4, 5);
    m.setBlock(0, 0, 1.0);
    m.setBlock(0, 1, 2.0);
    m.compress();

    // These go to btemp since the positions don't exist in CSR
    m.setBlock(1, 0, 5.0);
    m.setBlock(0, 3, 3.0);
    m.compress();

    EXPECT_NEAR(m.block(0, 0), 1.0, kTol);
    EXPECT_NEAR(m.block(0, 1), 2.0, kTol);
    EXPECT_NEAR(m.block(0, 3), 3.0, kTol);
    EXPECT_NEAR(m.block(1, 0), 5.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, AutoCompressOnRead)
{
    CRS m(4, 4);
    m.setBlock(1, 2, 9.0);
    // No explicit compress(); block() auto-compresses
    EXPECT_NEAR(m.block(1, 2), 9.0, kTol);
    EXPECT_FALSE(m.getRowIndex().empty());
}

TEST(CompressedRowSparseMatrixGeneric, CountEmptyBlocks)
{
    CRS m(4, 4);
    m.setBlock(0, 0, 1.0);
    m.setBlock(0, 1, 2.0);
    m.setBlock(1, 0, 3.0);
    m.compress();

    EXPECT_EQ(m.countEmptyBlocks(), 0u);

    // Zero out one entry in-place (existing CSR entry)
    m.setBlock(0, 0, 0.0);
    EXPECT_EQ(m.countEmptyBlocks(), 1u);
}

// ==================== CSR Structure Access ====================

TEST(CompressedRowSparseMatrixGeneric, GetRowIndex)
{
    CRS m(6, 6);
    m.setBlock(0, 0, 1.0);
    m.setBlock(2, 1, 2.0);
    m.setBlock(5, 3, 3.0);
    m.compress();

    const auto& ri = m.getRowIndex();
    ASSERT_EQ(ri.size(), 3u);
    EXPECT_EQ(ri[0], 0);
    EXPECT_EQ(ri[1], 2);
    EXPECT_EQ(ri[2], 5);
}

TEST(CompressedRowSparseMatrixGeneric, GetRowBegin)
{
    CRS m(6, 6);
    m.setBlock(0, 0, 1.0);
    m.setBlock(0, 2, 2.0);
    m.setBlock(2, 1, 3.0);
    m.compress();

    const auto& rb = m.getRowBegin();
    ASSERT_EQ(rb.size(), 3u);
    EXPECT_EQ(rb[0], 0);
    EXPECT_EQ(rb[1], 2);
    EXPECT_EQ(rb[2], 3);
}

TEST(CompressedRowSparseMatrixGeneric, GetRowRange)
{
    CRS m(6, 6);
    m.setBlock(0, 0, 1.0);
    m.setBlock(0, 2, 2.0);
    m.setBlock(2, 1, 3.0);
    m.compress();

    auto r0 = m.getRowRange(0);
    EXPECT_EQ(r0.begin(), 0);
    EXPECT_EQ(r0.end(), 2);
    EXPECT_EQ(r0.size(), 2);

    auto r1 = m.getRowRange(1);
    EXPECT_EQ(r1.begin(), 2);
    EXPECT_EQ(r1.end(), 3);
    EXPECT_EQ(r1.size(), 1);

    auto r2 = m.getRowRange(2);
    EXPECT_TRUE(r2.isInvalid());
}

TEST(CompressedRowSparseMatrixGeneric, GetRowRangeEmpty)
{
    CRS m;
    auto r = m.getRowRange(0);
    EXPECT_TRUE(r.isInvalid());
}

TEST(CompressedRowSparseMatrixGeneric, GetColsIndexAndValue)
{
    CRS m(4, 5);
    m.setBlock(0, 1, 1.0);
    m.setBlock(0, 3, 2.0);
    m.setBlock(2, 0, 3.0);
    m.compress();

    const auto& ci = m.getColsIndex();
    const auto& cv = m.getColsValue();

    ASSERT_EQ(ci.size(), 3u);
    ASSERT_EQ(cv.size(), 3u);

    EXPECT_EQ(ci[0], 1);
    EXPECT_NEAR(cv[0], 1.0, kTol);
    EXPECT_EQ(ci[1], 3);
    EXPECT_NEAR(cv[1], 2.0, kTol);
    EXPECT_EQ(ci[2], 0);
    EXPECT_NEAR(cv[2], 3.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, SortedFind)
{
    CRS::VecIndex v;
    v.push_back(1);
    v.push_back(3);
    v.push_back(5);
    v.push_back(8);
    v.push_back(12);

    CRS::Index result = 0;

    EXPECT_TRUE(CRS::sortedFind(v, 5, result));
    EXPECT_EQ(result, 2);

    EXPECT_TRUE(CRS::sortedFind(v, 1, result));
    EXPECT_EQ(result, 0);

    EXPECT_TRUE(CRS::sortedFind(v, 12, result));
    EXPECT_EQ(result, 4);

    EXPECT_FALSE(CRS::sortedFind(v, 7, result));
    EXPECT_FALSE(CRS::sortedFind(v, 0, result));

    // Sub-range search
    result = 0;
    CRS::Range subRange(1, 4); // elements at indices 1..3: {3, 5, 8}
    EXPECT_TRUE(CRS::sortedFind(v, subRange, 5, result));
    EXPECT_EQ(result, 2);
    EXPECT_FALSE(CRS::sortedFind(v, subRange, 1, result));
    EXPECT_FALSE(CRS::sortedFind(v, subRange, 12, result));
}

// ==================== Clearing Operations ====================

TEST(CompressedRowSparseMatrixGeneric, Clear)
{
    CRS m(4, 4);
    m.setBlock(0, 0, 1.0);
    m.setBlock(1, 2, 5.0);
    m.compress();
    EXPECT_FALSE(m.getColsValue().empty());

    m.clear();
    // ClearByZeros + CompressZeros: zeros all values then compresses them away
    EXPECT_TRUE(m.getColsValue().empty());
    EXPECT_TRUE(m.getRowIndex().empty());
}

TEST(CompressedRowSparseMatrixGeneric, ClearRowBlock)
{
    CRS m(4, 4);
    m.setBlock(0, 0, 1.0);
    m.setBlock(0, 1, 2.0);
    m.setBlock(1, 0, 3.0);
    m.setBlock(2, 2, 4.0);
    m.compress();

    m.clearRowBlock(0);

    EXPECT_NEAR(m.block(0, 0), 0.0, kTol);
    EXPECT_NEAR(m.block(0, 1), 0.0, kTol);
    EXPECT_NEAR(m.block(1, 0), 3.0, kTol);
    EXPECT_NEAR(m.block(2, 2), 4.0, kTol);
}

TEST(CompressedRowSparseMatrixGeneric, ClearColBlock)
{
    CRS m(4, 4);
    m.setBlock(0, 1, 1.0);
    m.setBlock(1, 1, 2.0);
    m.setBlock(2, 0, 3.0);
    m.setBlock(2, 1, 4.0);
    m.compress();

    m.clearColBlock(1);

    EXPECT_NEAR(m.block(0, 1), 0.0, kTol);
    EXPECT_NEAR(m.block(1, 1), 0.0, kTol);
    EXPECT_NEAR(m.block(2, 1), 0.0, kTol);
    EXPECT_NEAR(m.block(2, 0), 3.0, kTol);
}

// ==================== Row Operations ====================

TEST(CompressedRowSparseMatrixGeneric, FullRows)
{
    CRS m(6, 6);
    m.setBlock(0, 0, 1.0);
    m.setBlock(5, 1, 2.0);
    m.compress();

    EXPECT_EQ(m.getRowIndex().size(), 2u);

    m.fullRows();

    const auto& ri = m.getRowIndex();
    EXPECT_EQ(ri.size(), 6u);
    for (CRS::Index i = 0; i < 6; ++i)
        EXPECT_EQ(ri[i], i);
}

TEST(CompressedRowSparseMatrixGeneric, FullRowsEmpty)
{
    CRS m(4, 4);
    m.fullRows();
    EXPECT_EQ(m.getRowIndex().size(), 4u);
    EXPECT_TRUE(m.getColsValue().empty());
}

TEST(CompressedRowSparseMatrixGeneric, ShiftIndices)
{
    CRS m(4, 4);
    m.setBlock(0, 1, 1.0);
    m.setBlock(2, 3, 2.0);
    m.compress();

    const auto riBefore = m.getRowIndex();
    const auto rbBefore = m.getRowBegin();
    const auto ciBefore = m.getColsIndex();

    m.shiftIndices(1);

    const auto& ri = m.getRowIndex();
    const auto& rb = m.getRowBegin();
    const auto& ci = m.getColsIndex();

    for (std::size_t i = 0; i < ri.size(); ++i)
        EXPECT_EQ(ri[i], riBefore[i] + 1);
    for (std::size_t i = 0; i < rb.size(); ++i)
        EXPECT_EQ(rb[i], rbBefore[i] + 1);
    for (std::size_t i = 0; i < ci.size(); ++i)
        EXPECT_EQ(ci[i], ciBefore[i] + 1);
}

// ==================== Matrix Operations ====================

TEST(CompressedRowSparseMatrixGeneric, Mul)
{
    // A(3x2) * B(2x2) = C(3x2)
    CRS A(3, 2), B(2, 2), C;

    A.setBlock(0, 0, 1.0); A.setBlock(0, 1, 2.0);
    A.setBlock(1, 0, 3.0); A.setBlock(1, 1, 4.0);
    A.setBlock(2, 0, 5.0); A.setBlock(2, 1, 6.0);
    A.compress();

    B.setBlock(0, 0, 7.0); B.setBlock(0, 1, 8.0);
    B.setBlock(1, 0, 9.0); B.setBlock(1, 1, 10.0);
    B.compress();

    A.mul(C, B);

    EXPECT_EQ(C.rowBSize(), 3);
    EXPECT_EQ(C.colBSize(), 2);

    EXPECT_NEAR(C.block(0, 0), 25.0, kTol);  // 1*7 + 2*9
    EXPECT_NEAR(C.block(0, 1), 28.0, kTol);  // 1*8 + 2*10
    EXPECT_NEAR(C.block(1, 0), 57.0, kTol);  // 3*7 + 4*9
    EXPECT_NEAR(C.block(1, 1), 64.0, kTol);  // 3*8 + 4*10
    EXPECT_NEAR(C.block(2, 0), 89.0, kTol);  // 5*7 + 6*9
    EXPECT_NEAR(C.block(2, 1), 100.0, kTol); // 5*8 + 6*10
}

TEST(CompressedRowSparseMatrixGeneric, MulTranspose)
{
    // C = A^T(2x3) * B(3x2) = (2x2)
    CRS A(3, 2), B(3, 2), C;

    A.setBlock(0, 0, 1.0); A.setBlock(0, 1, 2.0);
    A.setBlock(1, 0, 3.0); A.setBlock(1, 1, 4.0);
    A.setBlock(2, 0, 5.0); A.setBlock(2, 1, 6.0);
    A.compress();

    B.setBlock(0, 0, 1.0); B.setBlock(0, 1, 2.0);
    B.setBlock(1, 0, 3.0); B.setBlock(1, 1, 4.0);
    B.setBlock(2, 0, 5.0); B.setBlock(2, 1, 6.0);
    B.compress();

    A.mulTranspose(C, B);

    EXPECT_EQ(C.rowBSize(), 2);
    EXPECT_EQ(C.colBSize(), 2);

    EXPECT_NEAR(C.block(0, 0), 35.0, kTol); // 1*1 + 3*3 + 5*5
    EXPECT_NEAR(C.block(0, 1), 44.0, kTol); // 1*2 + 3*4 + 5*6
    EXPECT_NEAR(C.block(1, 0), 44.0, kTol); // 2*1 + 4*3 + 6*5
    EXPECT_NEAR(C.block(1, 1), 56.0, kTol); // 2*2 + 4*4 + 6*6
}

TEST(CompressedRowSparseMatrixGeneric, TransposeFullRows)
{
    CRS A(3, 2), AT;

    A.setBlock(0, 0, 1.0); A.setBlock(0, 1, 2.0);
    A.setBlock(1, 0, 3.0); A.setBlock(1, 1, 4.0);
    A.setBlock(2, 0, 5.0);
    A.compress();
    A.fullRows();

    A.transposeFullRows(AT);

    EXPECT_EQ(AT.rowBSize(), 2);
    EXPECT_EQ(AT.colBSize(), 3);

    EXPECT_NEAR(AT.block(0, 0), 1.0, kTol);
    EXPECT_NEAR(AT.block(0, 1), 3.0, kTol);
    EXPECT_NEAR(AT.block(0, 2), 5.0, kTol);
    EXPECT_NEAR(AT.block(1, 0), 2.0, kTol);
    EXPECT_NEAR(AT.block(1, 1), 4.0, kTol);
    EXPECT_NEAR(AT.block(1, 2), 0.0, kTol);
}

// ==================== Swap ====================

TEST(CompressedRowSparseMatrixGeneric, Swap)
{
    CRS a(3, 3), b(4, 4);
    a.setBlock(0, 0, 1.0);
    a.compress();
    b.setBlock(2, 3, 7.0);
    b.compress();

    a.swap(b);

    EXPECT_EQ(a.rowBSize(), 4);
    EXPECT_EQ(a.colBSize(), 4);
    EXPECT_NEAR(a.block(2, 3), 7.0, kTol);

    EXPECT_EQ(b.rowBSize(), 3);
    EXPECT_EQ(b.colBSize(), 3);
    EXPECT_NEAR(b.block(0, 0), 1.0, kTol);
}

// ==================== Name ====================

TEST(CompressedRowSparseMatrixGeneric, Name)
{
    EXPECT_STREQ(CRS::Name(), "CompressedRowSparseMatrixd");
}

// ==================== Block Type Spot-Check (Mat3x3d) ====================

TEST(CompressedRowSparseMatrixGeneric, Mat3x3dInsertAndRetrieve)
{
    CRSMat3 m(2, 2);
    Mat3 block;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            block(i, j) = static_cast<double>(i * 3 + j + 1);

    m.setBlock(0, 1, block);
    const auto& retrieved = m.block(0, 1);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(retrieved(i, j), block(i, j), kTol);
}

TEST(CompressedRowSparseMatrixGeneric, Mat3x3dMul)
{
    // A(1x2 blocks) * B(2x1 blocks) = C(1x1 block)
    CRSMat3 A(1, 2), B(2, 1), C;

    Mat3 I;
    I.identity();

    Mat3 D;
    D.clear();
    for (int i = 0; i < 3; ++i) D(i, i) = 2.0;

    A.setBlock(0, 0, I);
    A.setBlock(0, 1, D);
    A.compress();

    B.setBlock(0, 0, I);
    B.setBlock(1, 0, I);
    B.compress();

    A.mul(C, B);

    // C(0,0) = I*I + D*I = I + D = diag(3,3,3)
    const auto& result = C.block(0, 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const double expected = (i == j) ? 3.0 : 0.0;
            EXPECT_NEAR(result(i, j), expected, kTol);
        }
}
