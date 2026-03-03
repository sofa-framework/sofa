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
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>
#include <sofa/linearalgebra/FullVector.h>
#include <gtest/gtest.h>

using CRSMech = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<double>;
using CRSMechMat3 = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<sofa::type::Mat<3, 3, double>>;
using FVec = sofa::linearalgebra::FullVector<double>;

constexpr double kTol = 1e-10;

// ==================== Construction & Sizing ====================

TEST(CompressedRowSparseMatrixMechanical, DefaultConstruction)
{
    CRSMech m;
    EXPECT_EQ(m.rowSize(), 0);
    EXPECT_EQ(m.colSize(), 0);
    EXPECT_EQ(m.rowBSize(), 0);
    EXPECT_EQ(m.colBSize(), 0);
    EXPECT_EQ(m.nRow, 0);
    EXPECT_EQ(m.nCol, 0);
}

TEST(CompressedRowSparseMatrixMechanical, ConstructWithScalarDimensions)
{
    CRSMech m(6, 8);
    EXPECT_EQ(m.rowSize(), 6);
    EXPECT_EQ(m.colSize(), 8);
    EXPECT_EQ(m.rowBSize(), 6);
    EXPECT_EQ(m.colBSize(), 8);
}

TEST(CompressedRowSparseMatrixMechanical, Resize)
{
    CRSMech m(10, 10);
    m.set(0, 0, 5.0);
    m.compress();

    m.resize(5, 7);
    EXPECT_EQ(m.rowSize(), 5);
    EXPECT_EQ(m.colSize(), 7);
    EXPECT_EQ(m.rowBSize(), 5);
    EXPECT_EQ(m.colBSize(), 7);
}

TEST(CompressedRowSparseMatrixMechanical, ResizeBlock)
{
    CRSMech m;
    m.resizeBlock(3, 4);
    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 4);
    EXPECT_EQ(m.nRow, 3);
    EXPECT_EQ(m.nCol, 4);
}

TEST(CompressedRowSparseMatrixMechanical, Extend)
{
    CRSMech m(4, 4);
    m.set(0, 0, 1.0);
    m.set(1, 1, 2.0);
    m.compress();

    m.extend(10, 10);
    EXPECT_EQ(m.rowSize(), 10);
    EXPECT_EQ(m.colSize(), 10);
    EXPECT_EQ(m.rowBSize(), 10);
    EXPECT_EQ(m.colBSize(), 10);

    // data is preserved
    EXPECT_NEAR(m.element(0, 0), 1.0, kTol);
    EXPECT_NEAR(m.element(1, 1), 2.0, kTol);
}

// ==================== Scalar Element Access ====================

TEST(CompressedRowSparseMatrixMechanical, SetAndElement)
{
    CRSMech m(5, 5);
    m.set(2, 3, 7.5);
    EXPECT_NEAR(m.element(2, 3), 7.5, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, AddScalar)
{
    CRSMech m(5, 5);
    m.set(0, 0, 3.0);
    m.compress();
    m.add(0, 0, 2.0);
    EXPECT_NEAR(m.element(0, 0), 5.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, ClearScalar)
{
    CRSMech m(5, 5);
    m.set(1, 2, 5.0);
    m.compress();
    m.clear(1, 2);
    EXPECT_NEAR(m.element(1, 2), 0.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, ElementNonExistent)
{
    CRSMech m(5, 5);
    m.set(0, 0, 1.0);
    m.compress();
    EXPECT_NEAR(m.element(3, 4), 0.0, kTol);
}

// ==================== Row/Column Clear ====================

TEST(CompressedRowSparseMatrixMechanical, ClearRow)
{
    CRSMech m(4, 4);
    m.set(1, 0, 1.0);
    m.set(1, 1, 2.0);
    m.set(1, 2, 3.0);
    m.set(1, 3, 4.0);
    m.set(0, 0, 10.0);
    m.set(2, 2, 20.0);
    m.compress();

    m.clearRow(1);

    EXPECT_NEAR(m.element(1, 0), 0.0, kTol);
    EXPECT_NEAR(m.element(1, 1), 0.0, kTol);
    EXPECT_NEAR(m.element(1, 2), 0.0, kTol);
    EXPECT_NEAR(m.element(1, 3), 0.0, kTol);
    // other rows unaffected
    EXPECT_NEAR(m.element(0, 0), 10.0, kTol);
    EXPECT_NEAR(m.element(2, 2), 20.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, ClearCol)
{
    CRSMech m(4, 4);
    m.set(0, 2, 1.0);
    m.set(1, 2, 2.0);
    m.set(2, 2, 3.0);
    m.set(3, 2, 4.0);
    m.set(0, 0, 10.0);
    m.set(3, 3, 20.0);
    m.compress();

    m.clearCol(2);

    EXPECT_NEAR(m.element(0, 2), 0.0, kTol);
    EXPECT_NEAR(m.element(1, 2), 0.0, kTol);
    EXPECT_NEAR(m.element(2, 2), 0.0, kTol);
    EXPECT_NEAR(m.element(3, 2), 0.0, kTol);
    // other columns unaffected
    EXPECT_NEAR(m.element(0, 0), 10.0, kTol);
    EXPECT_NEAR(m.element(3, 3), 20.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, ClearRowCol)
{
    CRSMech m(4, 4);
    // set up a symmetric-looking matrix with fullDiagonal
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 2.0);
    m.set(1, 1, 3.0);
    m.set(1, 2, 4.0);
    m.set(2, 1, 4.0);
    m.set(2, 2, 5.0);
    m.set(3, 3, 6.0);
    m.compress();
    m.fullDiagonal();

    m.clearRowCol(1);

    // row 1 and column 1 should be zero
    for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(m.element(1, j), 0.0, kTol) << "element(1," << j << ")";
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(m.element(i, 1), 0.0, kTol) << "element(" << i << ",1)";

    // other entries unaffected
    EXPECT_NEAR(m.element(0, 0), 1.0, kTol);
    EXPECT_NEAR(m.element(2, 2), 5.0, kTol);
    EXPECT_NEAR(m.element(3, 3), 6.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, ClearOverride)
{
    CRSMech m(4, 4);
    m.set(0, 0, 1.0);
    m.set(1, 2, 5.0);
    m.set(2, 3, 7.0);
    m.compress();

    const auto numValuesBefore = m.getColsValue().size();
    EXPECT_GT(numValuesBefore, 0u);

    m.clear();

    // ClearByZeros=true zeros values, CompressZeros=false preserves structure
    EXPECT_EQ(m.getColsValue().size(), numValuesBefore);
    for (const auto& v : m.getColsValue())
    {
        EXPECT_NEAR(v, 0.0, kTol);
    }
}

// ==================== Diagonal ====================

TEST(CompressedRowSparseMatrixMechanical, FullDiagonal)
{
    CRSMech m(4, 4);
    m.set(0, 1, 2.0);
    m.set(2, 3, 5.0);
    m.compress();

    m.fullDiagonal();

    const auto& ri = m.getRowIndex();
    ASSERT_EQ(ri.size(), 4u);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(ri[i], i);

    // diagonal entries exist (value is 0 where not explicitly set)
    EXPECT_NEAR(m.element(0, 0), 0.0, kTol);
    EXPECT_NEAR(m.element(1, 1), 0.0, kTol);
    EXPECT_NEAR(m.element(2, 2), 0.0, kTol);
    EXPECT_NEAR(m.element(3, 3), 0.0, kTol);
    // off-diagonal preserved
    EXPECT_NEAR(m.element(0, 1), 2.0, kTol);
    EXPECT_NEAR(m.element(2, 3), 5.0, kTol);
}

// ==================== Matrix-Vector Products ====================

TEST(CompressedRowSparseMatrixMechanical, MulVector)
{
    // M = [1 2 0]
    //     [0 3 4]
    //     [5 0 6]
    CRSMech m(3, 3);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0);
    m.set(1, 1, 3.0); m.set(1, 2, 4.0);
    m.set(2, 0, 5.0); m.set(2, 2, 6.0);
    m.compress();

    FVec v(3);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0;

    FVec res(3);
    m.mul(res, v);

    // M*v = [1+4, 6+12, 5+18] = [5, 18, 23]
    EXPECT_NEAR(res[0], 5.0, kTol);
    EXPECT_NEAR(res[1], 18.0, kTol);
    EXPECT_NEAR(res[2], 23.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, AddMulVector)
{
    // M = [2 1]
    //     [3 0]
    CRSMech m(2, 2);
    m.set(0, 0, 2.0); m.set(0, 1, 1.0);
    m.set(1, 0, 3.0);
    m.compress();

    FVec v(2);
    v[0] = 1.0; v[1] = 2.0;

    FVec res(2);
    res[0] = 0.0; res[1] = 0.0;
    m.addMul(res, v);

    // res = M * v = [2+2, 3+0] = [4, 3]
    EXPECT_NEAR(res[0], 4.0, kTol);
    EXPECT_NEAR(res[1], 3.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, AddMultTranspose)
{
    // For CRSMechanicalPolicy, IsAlwaysSymmetric=true so addMultTranspose
    // delegates to addMul (this^T == this for a symmetric matrix).
    // Use a symmetric matrix to verify.
    //
    // M = [1 2]
    //     [2 3]
    // v = [1, 2], expected M^T * v = M * v = [1+4, 2+6] = [5, 8]
    CRSMech m(2, 2);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0);
    m.set(1, 0, 2.0); m.set(1, 1, 3.0);
    m.compress();

    FVec v(2);
    v[0] = 1.0; v[1] = 2.0;

    FVec res(2);
    res[0] = 0.0; res[1] = 0.0;
    m.addMultTranspose(res, v);

    EXPECT_NEAR(res[0], 5.0, kTol);
    EXPECT_NEAR(res[1], 8.0, kTol);
}

// ==================== Arithmetic Operators ====================

TEST(CompressedRowSparseMatrixMechanical, OperatorPlus)
{
    CRSMech A(3, 3), B(3, 3);
    A.set(0, 0, 1.0); A.set(0, 1, 2.0); A.set(1, 1, 3.0);
    B.set(0, 0, 4.0); B.set(1, 0, 1.0); B.set(1, 1, 2.0);
    A.compress();
    B.compress();

    CRSMech C = A + B;
    EXPECT_NEAR(C.element(0, 0), 5.0, kTol);
    EXPECT_NEAR(C.element(0, 1), 2.0, kTol);
    EXPECT_NEAR(C.element(1, 0), 1.0, kTol);
    EXPECT_NEAR(C.element(1, 1), 5.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, OperatorPlusEquals)
{
    CRSMech A(3, 3), B(3, 3);
    A.set(0, 0, 1.0); A.set(1, 1, 2.0);
    B.set(0, 0, 3.0); B.set(1, 1, 4.0); B.set(2, 2, 5.0);
    A.compress();
    B.compress();

    A += B;
    EXPECT_NEAR(A.element(0, 0), 4.0, kTol);
    EXPECT_NEAR(A.element(1, 1), 6.0, kTol);
    EXPECT_NEAR(A.element(2, 2), 5.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, OperatorMinusEquals)
{
    CRSMech A(3, 3), B(3, 3);
    A.set(0, 0, 5.0); A.set(1, 1, 8.0);
    B.set(0, 0, 2.0); B.set(1, 1, 3.0);
    A.compress();
    B.compress();

    A -= B;
    EXPECT_NEAR(A.element(0, 0), 3.0, kTol);
    EXPECT_NEAR(A.element(1, 1), 5.0, kTol);
}

// ==================== Info Methods ====================

TEST(CompressedRowSparseMatrixMechanical, GetCategory)
{
    CRSMech m;
    EXPECT_EQ(m.getCategory(), sofa::linearalgebra::BaseMatrix::MATRIX_SPARSE);
}

TEST(CompressedRowSparseMatrixMechanical, GetBlockRowsCols)
{
    CRSMech mScalar;
    EXPECT_EQ(mScalar.getBlockRows(), 1);
    EXPECT_EQ(mScalar.getBlockCols(), 1);

    CRSMechMat3 mBlock;
    EXPECT_EQ(mBlock.getBlockRows(), 3);
    EXPECT_EQ(mBlock.getBlockCols(), 3);
}

TEST(CompressedRowSparseMatrixMechanical, GetBandWidth)
{
    CRSMech mScalar;
    EXPECT_EQ(mScalar.getBandWidth(), 0); // NC - 1 = 0

    CRSMechMat3 mBlock;
    EXPECT_EQ(mBlock.getBandWidth(), 2); // NC - 1 = 2
}

TEST(CompressedRowSparseMatrixMechanical, RowColSize)
{
    CRSMech m(6, 8);
    EXPECT_EQ(m.rowSize(), 6);
    EXPECT_EQ(m.colSize(), 8);
    EXPECT_EQ(m.bRowSize(), 6);
    EXPECT_EQ(m.bColSize(), 8);
}

// ==================== Swap ====================

TEST(CompressedRowSparseMatrixMechanical, Swap)
{
    CRSMech a(4, 4), b(6, 6);
    a.set(0, 0, 1.0);
    a.set(1, 1, 2.0);
    a.compress();

    b.set(3, 3, 7.0);
    b.set(5, 5, 9.0);
    b.compress();

    a.swap(b);

    EXPECT_EQ(a.rowSize(), 6);
    EXPECT_EQ(a.colSize(), 6);
    EXPECT_NEAR(a.element(3, 3), 7.0, kTol);
    EXPECT_NEAR(a.element(5, 5), 9.0, kTol);

    EXPECT_EQ(b.rowSize(), 4);
    EXPECT_EQ(b.colSize(), 4);
    EXPECT_NEAR(b.element(0, 0), 1.0, kTol);
    EXPECT_NEAR(b.element(1, 1), 2.0, kTol);
}

// ==================== Filtering ====================

TEST(CompressedRowSparseMatrixMechanical, CopyUpper)
{
    CRSMech src(3, 3);
    src.set(0, 0, 1.0); src.set(0, 1, 2.0); src.set(0, 2, 3.0);
    src.set(1, 0, 4.0); src.set(1, 1, 5.0); src.set(1, 2, 6.0);
    src.set(2, 0, 7.0); src.set(2, 1, 8.0); src.set(2, 2, 9.0);
    src.compress();

    CRSMech dst;
    dst.copyUpper(src);

    // upper triangle: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    EXPECT_NEAR(dst.element(0, 0), 1.0, kTol);
    EXPECT_NEAR(dst.element(0, 1), 2.0, kTol);
    EXPECT_NEAR(dst.element(0, 2), 3.0, kTol);
    EXPECT_NEAR(dst.element(1, 1), 5.0, kTol);
    EXPECT_NEAR(dst.element(1, 2), 6.0, kTol);
    EXPECT_NEAR(dst.element(2, 2), 9.0, kTol);

    // lower triangle should not be present
    EXPECT_NEAR(dst.element(1, 0), 0.0, kTol);
    EXPECT_NEAR(dst.element(2, 0), 0.0, kTol);
    EXPECT_NEAR(dst.element(2, 1), 0.0, kTol);
}

TEST(CompressedRowSparseMatrixMechanical, CopyLower)
{
    CRSMech src(3, 3);
    src.set(0, 0, 1.0); src.set(0, 1, 2.0); src.set(0, 2, 3.0);
    src.set(1, 0, 4.0); src.set(1, 1, 5.0); src.set(1, 2, 6.0);
    src.set(2, 0, 7.0); src.set(2, 1, 8.0); src.set(2, 2, 9.0);
    src.compress();

    CRSMech dst;
    dst.copyLower(src);

    // lower triangle: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
    EXPECT_NEAR(dst.element(0, 0), 1.0, kTol);
    EXPECT_NEAR(dst.element(1, 0), 4.0, kTol);
    EXPECT_NEAR(dst.element(1, 1), 5.0, kTol);
    EXPECT_NEAR(dst.element(2, 0), 7.0, kTol);
    EXPECT_NEAR(dst.element(2, 1), 8.0, kTol);
    EXPECT_NEAR(dst.element(2, 2), 9.0, kTol);

    // upper triangle should not be present
    EXPECT_NEAR(dst.element(0, 1), 0.0, kTol);
    EXPECT_NEAR(dst.element(0, 2), 0.0, kTol);
    EXPECT_NEAR(dst.element(1, 2), 0.0, kTol);
}

// ==================== Mat3x3d Block Spot-Check ====================

TEST(CompressedRowSparseMatrixMechanical, Mat3x3dScalarAccess)
{
    CRSMechMat3 m(9, 9);

    // set scalar entries within the block at block position (0, 0)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m.set(i, j, static_cast<double>(i * 3 + j + 1));

    m.compress();

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m.element(i, j), static_cast<double>(i * 3 + j + 1), kTol)
                << "element(" << i << "," << j << ")";
}

TEST(CompressedRowSparseMatrixMechanical, Mat3x3dResize)
{
    CRSMechMat3 m(9, 9);
    EXPECT_EQ(m.rowSize(), 9);
    EXPECT_EQ(m.colSize(), 9);
    EXPECT_EQ(m.rowBSize(), 3);
    EXPECT_EQ(m.colBSize(), 3);
}

TEST(CompressedRowSparseMatrixMechanical, Mat3x3dMulVector)
{
    // Single 3x3 block = diagonal [1, 2, 3]
    CRSMechMat3 m(3, 3);
    m.set(0, 0, 1.0);
    m.set(1, 1, 2.0);
    m.set(2, 2, 3.0);
    m.compress();

    FVec v(3);
    v[0] = 1.0; v[1] = 1.0; v[2] = 1.0;

    FVec res(3);
    m.mul(res, v);

    EXPECT_NEAR(res[0], 1.0, kTol);
    EXPECT_NEAR(res[1], 2.0, kTol);
    EXPECT_NEAR(res[2], 3.0, kTol);
}
