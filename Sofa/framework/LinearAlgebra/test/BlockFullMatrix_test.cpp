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
#include <sofa/linearalgebra/BlockFullMatrix.inl>
#include <sofa/linearalgebra/FullVector.h>

#include <gtest/gtest.h>

#include <cmath>

namespace sofa
{

using BFM2 = linearalgebra::BlockFullMatrix<2, double>;
using Block2 = BFM2::Block;
using Vec2 = type::Vec<2, double>;

constexpr double tol = 1e-10;

// ============================================================
// Construction & Dimensions
// ============================================================

TEST(BlockFullMatrix, DefaultConstruction)
{
    BFM2 m;
    EXPECT_EQ(m.rowSize(), 0);
    EXPECT_EQ(m.colSize(), 0);
    EXPECT_EQ(m.ptr(), nullptr);
}

TEST(BlockFullMatrix, ConstructWithDimensions)
{
    BFM2 m(4, 4);
    EXPECT_EQ(m.rowSize(), 4);
    EXPECT_EQ(m.colSize(), 4);
    EXPECT_NE(m.ptr(), nullptr);
}

TEST(BlockFullMatrix, Resize)
{
    // Use nBCol >= 4 so that clear() (which iterates 3*nBRow) doesn't go OOB.
    // With BSIZE=2, 8 cols → nBCol=4.
    BFM2 m;
    m.resize(4, 8);
    EXPECT_EQ(m.rowSize(), 4);
    EXPECT_EQ(m.colSize(), 8);
    EXPECT_NE(m.ptr(), nullptr);

    m.resize(6, 8);
    EXPECT_EQ(m.rowSize(), 6);
    EXPECT_EQ(m.colSize(), 8);
}

TEST(BlockFullMatrix, ResizeClearsData)
{
    // Use 4x8 (nBRow=2, nBCol=4) so clear() under-clears rather than OOB.
    // clear() iterates 3*nBRow=6 blocks out of nBRow*nBCol=8.
    // Blocks at indices 6,7 (block row 1, columns 2-3) won't be cleared.
    BFM2 m(4, 8);
    m.set(0, 0, 7.0);
    m.set(3, 7, 9.0);  // block (1,3) = data[7], won't be cleared by buggy clear()

    m.resize(4, 8);

    for (SignedIndex i = 0; i < 4; ++i)
        for (SignedIndex j = 0; j < 8; ++j)
            EXPECT_NEAR(m.element(i, j), 0.0, tol)
                << "element(" << i << "," << j << ") not zero after resize";
}

// ============================================================
// Element Access (set / element / add / clear)
// ============================================================

TEST(BlockFullMatrix, SetAndElement)
{
    BFM2 m(4, 4);

    // element in block (0,0)
    m.set(0, 1, 1.5);
    EXPECT_NEAR(m.element(0, 1), 1.5, tol);

    // element in block (1,0)
    m.set(2, 0, -3.0);
    EXPECT_NEAR(m.element(2, 0), -3.0, tol);

    // element in block (0,1)
    m.set(1, 3, 4.25);
    EXPECT_NEAR(m.element(1, 3), 4.25, tol);

    // element in block (1,1)
    m.set(3, 3, 100.0);
    EXPECT_NEAR(m.element(3, 3), 100.0, tol);
}

TEST(BlockFullMatrix, Add)
{
    BFM2 m(4, 4);
    m.set(0, 0, 3.0);
    m.add(0, 0, 2.0);
    EXPECT_NEAR(m.element(0, 0), 5.0, tol);
}

TEST(BlockFullMatrix, ClearElement)
{
    BFM2 m(4, 4);
    m.set(1, 1, 5.0);
    m.set(1, 0, 2.0);
    m.clear(1, 1);
    EXPECT_NEAR(m.element(1, 1), 0.0, tol);
    EXPECT_NEAR(m.element(1, 0), 2.0, tol);
}

// ============================================================
// Block Access
// ============================================================

TEST(BlockFullMatrix, BlocAccess)
{
    BFM2 m(4, 4);

    // Write through block reference
    Block2& blk = m.bloc(0, 1);
    blk(0, 0) = 10.0;
    blk(1, 1) = 20.0;

    // Read via element()
    EXPECT_NEAR(m.element(0, 2), 10.0, tol);
    EXPECT_NEAR(m.element(1, 3), 20.0, tol);
}

TEST(BlockFullMatrix, SubAndAsub)
{
    BFM2 m(4, 4);
    m.set(2, 2, 42.0);

    // sub() takes element indices, returns block reference
    const Block2& s = m.sub(2, 2, 2, 2);
    EXPECT_NEAR(s.element(0, 0), 42.0, tol);

    // asub() takes block indices
    const Block2& a = m.asub(1, 1, 2, 2);
    EXPECT_NEAR(a.element(0, 0), 42.0, tol);
}

TEST(BlockFullMatrix, GetSetSubMatrix)
{
    BFM2 m(4, 4);
    Block2 input;
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    input.set(1, 0, 3.0);
    input.set(1, 1, 4.0);

    m.setSubMatrix(2, 2, 2, 2, input);

    Block2 output;
    m.getSubMatrix(2, 2, 2, 2, output);

    for (SignedIndex i = 0; i < 2; ++i)
        for (SignedIndex j = 0; j < 2; ++j)
            EXPECT_NEAR(output.element(i, j), input.element(i, j), tol);
}

// ============================================================
// Clearing Operations
// ============================================================

TEST(BlockFullMatrix, ClearAll)
{
    // 4x8 matrix with BSIZE=2 => nBRow=2, nBCol=4.
    // clear() iterates 3*nBRow=6 instead of nBRow*nBCol=8, so the last 2 blocks
    // (block row 1, columns 2-3) are NOT cleared. This exposes the bug without
    // causing an out-of-bounds write (which would happen with nBCol < 3).
    BFM2 m(4, 8);
    m.set(0, 0, 1.0);   // block (0,0) = data[0]
    m.set(0, 2, 2.0);   // block (0,1) = data[1]
    m.set(0, 4, 3.0);   // block (0,2) = data[2]
    m.set(0, 6, 4.0);   // block (0,3) = data[3]
    m.set(2, 0, 5.0);   // block (1,0) = data[4]
    m.set(2, 2, 6.0);   // block (1,1) = data[5]
    m.set(2, 4, 7.0);   // block (1,2) = data[6] — NOT cleared by buggy code
    m.set(2, 6, 8.0);   // block (1,3) = data[7] — NOT cleared by buggy code

    m.clear();

    for (SignedIndex i = 0; i < 4; ++i)
        for (SignedIndex j = 0; j < 8; ++j)
            EXPECT_NEAR(m.element(i, j), 0.0, tol)
                << "element(" << i << "," << j << ") not zero after clear()";
}

TEST(BlockFullMatrix, ClearRow)
{
    BFM2 m(4, 4);
    for (SignedIndex i = 0; i < 4; ++i)
        for (SignedIndex j = 0; j < 4; ++j)
            m.set(i, j, static_cast<double>(i * 4 + j + 1));

    m.clearRow(1);

    for (SignedIndex j = 0; j < 4; ++j)
        EXPECT_NEAR(m.element(1, j), 0.0, tol)
            << "row 1, col " << j << " should be zero";

    // Other rows unchanged
    EXPECT_NEAR(m.element(0, 0), 1.0, tol);
    EXPECT_NEAR(m.element(2, 2), 11.0, tol);
    EXPECT_NEAR(m.element(3, 3), 16.0, tol);
}

TEST(BlockFullMatrix, ClearCol)
{
    BFM2 m(4, 4);
    for (SignedIndex i = 0; i < 4; ++i)
        for (SignedIndex j = 0; j < 4; ++j)
            m.set(i, j, static_cast<double>(i * 4 + j + 1));

    m.clearCol(2);

    for (SignedIndex i = 0; i < 4; ++i)
        EXPECT_NEAR(m.element(i, 2), 0.0, tol)
            << "row " << i << ", col 2 should be zero";

    // Other columns unchanged
    EXPECT_NEAR(m.element(0, 0), 1.0, tol);
    EXPECT_NEAR(m.element(1, 1), 6.0, tol);
    EXPECT_NEAR(m.element(3, 3), 16.0, tol);
}

TEST(BlockFullMatrix, ClearRowCol)
{
    BFM2 m(4, 4);
    for (SignedIndex i = 0; i < 4; ++i)
        for (SignedIndex j = 0; j < 4; ++j)
            m.set(i, j, static_cast<double>(i * 4 + j + 1));

    m.clearRowCol(1);

    // Row 1 all zeros
    for (SignedIndex j = 0; j < 4; ++j)
        EXPECT_NEAR(m.element(1, j), 0.0, tol);

    // Col 1 all zeros
    for (SignedIndex i = 0; i < 4; ++i)
        EXPECT_NEAR(m.element(i, 1), 0.0, tol);

    // Other elements unchanged (except those on row/col 1)
    EXPECT_NEAR(m.element(0, 0), 1.0, tol);
    EXPECT_NEAR(m.element(2, 2), 11.0, tol);
    EXPECT_NEAR(m.element(3, 3), 16.0, tol);
}

// ============================================================
// Matrix-Vector Multiplication
// ============================================================

TEST(BlockFullMatrix, IdentityTimesVector)
{
    BFM2 m(4, 4);
    // Set to identity
    for (SignedIndex i = 0; i < 4; ++i)
        m.set(i, i, 1.0);

    linearalgebra::FullVector<double> v(4);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0; v[3] = 4.0;

    linearalgebra::FullVector<double> res = m * v;

    for (SignedIndex i = 0; i < 4; ++i)
        EXPECT_NEAR(res[i], v[i], tol);
}

TEST(BlockFullMatrix, GeneralProduct)
{
    // 4x4 matrix, BSIZE=2
    // M = [1 2 0 0]     v = [1]
    //     [3 4 0 0]         [2]
    //     [0 0 5 6]         [3]
    //     [0 0 7 8]         [4]
    //
    // M*v = [1*1+2*2, 3*1+4*2, 5*3+6*4, 7*3+8*4] = [5, 11, 39, 53]
    BFM2 m(4, 4);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0);
    m.set(1, 0, 3.0); m.set(1, 1, 4.0);
    m.set(2, 2, 5.0); m.set(2, 3, 6.0);
    m.set(3, 2, 7.0); m.set(3, 3, 8.0);

    linearalgebra::FullVector<double> v(4);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0; v[3] = 4.0;

    linearalgebra::FullVector<double> res = m * v;

    EXPECT_NEAR(res[0], 5.0, tol);
    EXPECT_NEAR(res[1], 11.0, tol);
    EXPECT_NEAR(res[2], 39.0, tol);
    EXPECT_NEAR(res[3], 53.0, tol);
}

TEST(BlockFullMatrix, SingleBlockColumn)
{
    // Matrix with nBCol=1: 4 rows, 2 cols (BSIZE=2 => nBRow=2, nBCol=1)
    // This tests the split-loop boundary in operator*
    // M = [1 2]     v = [3]
    //     [3 4]         [5]
    //     [5 6]
    //     [7 8]
    //
    // M*v = [1*3+2*5, 3*3+4*5, 5*3+6*5, 7*3+8*5] = [13, 29, 45, 61]
    BFM2 m(4, 2);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0);
    m.set(1, 0, 3.0); m.set(1, 1, 4.0);
    m.set(2, 0, 5.0); m.set(2, 1, 6.0);
    m.set(3, 0, 7.0); m.set(3, 1, 8.0);

    linearalgebra::FullVector<double> v(2);
    v[0] = 3.0; v[1] = 5.0;

    linearalgebra::FullVector<double> res = m * v;

    EXPECT_NEAR(res[0], 13.0, tol);
    EXPECT_NEAR(res[1], 29.0, tol);
    EXPECT_NEAR(res[2], 45.0, tol);
    EXPECT_NEAR(res[3], 61.0, tol);
}

// ============================================================
// Block Nested Class
// ============================================================

TEST(BlockFullMatrix, BlockNrowsNcols)
{
    Block2 b;
    EXPECT_EQ(b.Nrows(), 2);
    EXPECT_EQ(b.Ncols(), 2);
}

TEST(BlockFullMatrix, BlockSetAddElement)
{
    Block2 b;
    b.clear();
    b.set(0, 0, 3.0);
    EXPECT_NEAR(b.element(0, 0), 3.0, tol);

    b.add(0, 0, 2.0);
    EXPECT_NEAR(b.element(0, 0), 5.0, tol);

    b.set(1, 0, -1.0);
    EXPECT_NEAR(b.element(1, 0), -1.0, tol);
}

TEST(BlockFullMatrix, BlockTranspose)
{
    Block2 b;
    b.clear();
    b.set(0, 0, 1.0); b.set(0, 1, 2.0);
    b.set(1, 0, 3.0); b.set(1, 1, 4.0);

    auto tb = b.t();

    // tb * v should compute b^T * v
    // b^T = [1 3]    v = [1]   => [1*1+3*2, 2*1+4*2] = [7, 10]
    //       [2 4]        [2]
    Vec2 v(1.0, 2.0);
    Vec2 r = tb * v;

    EXPECT_NEAR(r[0], 7.0, tol);
    EXPECT_NEAR(r[1], 10.0, tol);
}

TEST(BlockFullMatrix, BlockInverse)
{
    Block2 b;
    b.clear();
    b.set(0, 0, 4.0); b.set(0, 1, 7.0);
    b.set(1, 0, 2.0); b.set(1, 1, 6.0);

    Block2 inv = b.i();

    // b * inv should be identity
    auto product = b * inv;
    EXPECT_NEAR(product(0, 0), 1.0, tol);
    EXPECT_NEAR(product(0, 1), 0.0, tol);
    EXPECT_NEAR(product(1, 0), 0.0, tol);
    EXPECT_NEAR(product(1, 1), 1.0, tol);
}

TEST(BlockFullMatrix, BlockNegation)
{
    Block2 b;
    b.clear();
    b.set(0, 0, 1.0); b.set(0, 1, -2.0);
    b.set(1, 0, 3.0); b.set(1, 1, -4.0);

    auto neg = -b;

    EXPECT_NEAR(neg(0, 0), -1.0, tol);
    EXPECT_NEAR(neg(0, 1), 2.0, tol);
    EXPECT_NEAR(neg(1, 0), -3.0, tol);
    EXPECT_NEAR(neg(1, 1), 4.0, tol);
}

TEST(BlockFullMatrix, BlockSubtraction)
{
    Block2 a;
    a.clear();
    a.set(0, 0, 5.0); a.set(0, 1, 6.0);
    a.set(1, 0, 7.0); a.set(1, 1, 8.0);

    Block2 b;
    b.clear();
    b.set(0, 0, 1.0); b.set(0, 1, 2.0);
    b.set(1, 0, 3.0); b.set(1, 1, 4.0);

    auto diff = a - b;

    EXPECT_NEAR(diff(0, 0), 4.0, tol);
    EXPECT_NEAR(diff(0, 1), 4.0, tol);
    EXPECT_NEAR(diff(1, 0), 4.0, tol);
    EXPECT_NEAR(diff(1, 1), 4.0, tol);
}

TEST(BlockFullMatrix, BlockMatVec)
{
    Block2 b;
    b.clear();
    b.set(0, 0, 1.0); b.set(0, 1, 2.0);
    b.set(1, 0, 3.0); b.set(1, 1, 4.0);

    Vec2 v(5.0, 6.0);
    Vec2 r = b * v;

    // [1 2] * [5] = [1*5+2*6, 3*5+4*6] = [17, 39]
    // [3 4]   [6]
    EXPECT_NEAR(r[0], 17.0, tol);
    EXPECT_NEAR(r[1], 39.0, tol);
}

// ============================================================
// Name
// ============================================================

TEST(BlockFullMatrix, Name)
{
    std::string name = BFM2::Name();
    EXPECT_EQ(name, "BlockFullMatrix2d");
}

// ============================================================
// Spot-Check with Production Type (BlockFullMatrix<6, SReal>)
// ============================================================

TEST(BlockFullMatrix, ProductionType)
{
    using BFM6 = linearalgebra::BlockFullMatrix<6, SReal>;

    BFM6 m(12, 12);
    EXPECT_EQ(m.rowSize(), 12);
    EXPECT_EQ(m.colSize(), 12);

    m.set(0, 0, 1.0);
    m.set(5, 5, 2.0);
    m.set(6, 6, 3.0);
    m.set(11, 11, 4.0);
    EXPECT_NEAR(m.element(0, 0), 1.0, tol);
    EXPECT_NEAR(m.element(5, 5), 2.0, tol);
    EXPECT_NEAR(m.element(6, 6), 3.0, tol);
    EXPECT_NEAR(m.element(11, 11), 4.0, tol);

    // Identity matrix-vector product
    linearalgebra::FullVector<SReal> v(12);
    for (SignedIndex i = 0; i < 12; ++i)
    {
        m.set(i, i, 1.0);
        v[i] = static_cast<SReal>(i + 1);
    }

    linearalgebra::FullVector<SReal> res = m * v;
    for (SignedIndex i = 0; i < 12; ++i)
        EXPECT_NEAR(res[i], v[i], tol);
}

} // namespace sofa
