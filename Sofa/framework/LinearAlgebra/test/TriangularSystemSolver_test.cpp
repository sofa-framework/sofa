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
#include <sofa/linearalgebra/TriangularSystemSolver.h>
#include <gtest/gtest.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/testing/NumericTest.h>


namespace sofa
{

TEST(TriangularSystemSolver, empty)
{
    const SReal* rightHandSideVector { nullptr };
    SReal* solution { nullptr };
    const sofa::Size* const L_columns { nullptr };
    const sofa::Size* const L_row { nullptr };
    const SReal* const L_values { nullptr };
    EXPECT_NO_THROW(
        sofa::linearalgebra::solveLowerUnitriangularSystemCSR(0, rightHandSideVector, solution, L_columns, L_row, L_values);
    );
}

TEST(TriangularSystemSolver, lower2x2)
{
    constexpr std::array<SReal, 2> rightHandSideVector { 6, 8 };
    std::array<SReal, 2> solution {};

    /**
     * [ 1 0 ]  *  [  6 ]  =  [ 6 ]
     * [ 2 1 ]  *  [ -4 ]  =  [ 8 ]
     */

    // CSR format
    constexpr std::array<sofa::Size, 3> L_columns { 0, 0, 1 };
    constexpr std::array<sofa::Size, 3> L_row { 0, 1, 3 };
    constexpr std::array<SReal, 3> L_values { 1, 2, 1 };

    sofa::linearalgebra::solveLowerUnitriangularSystemCSR(2, rightHandSideVector.data(), solution.data(), L_row.data(), L_columns.data(), L_values.data());
    EXPECT_FLOATINGPOINT_EQ(solution[0], static_cast<SReal>(6))
    EXPECT_FLOATINGPOINT_EQ(solution[1], static_cast<SReal>(8 - 2 * 6))
}

TEST(TriangularSystemSolver, lower3x3)
{
    constexpr std::array<SReal, 3> rightHandSideVector { 5, -9, 3 };
    std::array<SReal, 3> solution {};

    /**
     * [ 1 0 0 ]  *  [   5 ]  =  [  5 ]
     * [ 2 1 0 ]  *  [ -19 ]  =  [ -9 ]
     * [ 3 4 1 ]  *  [  64 ]  =  [  3 ]
     */

    // CSR format
    constexpr std::array<sofa::Size, 6> L_columns { 0, 0, 1, 0, 1, 2 };
    constexpr std::array<sofa::Size, 4> L_row { 0, 1, 3, 6 };
    constexpr std::array<SReal, 6> L_values { 1, 2, 1, 3, 4, 1 };

    sofa::linearalgebra::solveLowerUnitriangularSystemCSR(3, rightHandSideVector.data(), solution.data(), L_row.data(), L_columns.data(), L_values.data());
    EXPECT_FLOATINGPOINT_EQ(solution[0], static_cast<SReal>(5))
    EXPECT_FLOATINGPOINT_EQ(solution[1], static_cast<SReal>(-9 - 2 * 5))
    EXPECT_FLOATINGPOINT_EQ(solution[2], static_cast<SReal>(64))
}


TEST(TriangularSystemSolver, upper2x2)
{
    constexpr std::array<SReal, 2> rightHandSideVector { 6, 8 };
    std::array<SReal, 2> solution {};

    /**
     * [ 1 2 ]  *  [ -10 ]  =  [ 6 ]
     * [ 0 1 ]  *  [   8 ]  =  [ 8 ]
     */

    // CSR format
    constexpr std::array<sofa::Size, 3> L_columns { 0, 1, 1 };
    constexpr std::array<sofa::Size, 3> L_row { 0, 2, 3 };
    constexpr std::array<SReal, 3> L_values { 1, 2, 1 };

    sofa::linearalgebra::solveUpperUnitriangularSystemCSR(2, rightHandSideVector.data(), solution.data(), L_row.data(), L_columns.data(), L_values.data());
    EXPECT_FLOATINGPOINT_EQ(solution[1], static_cast<SReal>(8))
    EXPECT_FLOATINGPOINT_EQ(solution[0], static_cast<SReal>(-10))
}

TEST(TriangularSystemSolver, upper3x3)
{
    constexpr std::array<SReal, 3> rightHandSideVector { 5, -9, 3 };
    std::array<SReal, 3> solution {};

    /**
     * [ 1 2 3 ]  *  [  38 ]  =  [  5 ]
     * [ 0 1 4 ]  *  [ -21 ]  =  [ -9 ]
     * [ 0 0 1 ]  *  [   3 ]  =  [  3 ]
     */

    // CSR format
    constexpr std::array<sofa::Size, 6> L_columns { 0, 1, 2, 1, 2, 2 };
    constexpr std::array<sofa::Size, 4> L_row { 0, 3, 5, 6 };
    constexpr std::array<SReal, 6> L_values { 1, 2, 3, 1, 4, 1 };

    sofa::linearalgebra::solveUpperUnitriangularSystemCSR(3, rightHandSideVector.data(), solution.data(), L_row.data(), L_columns.data(), L_values.data());
    EXPECT_FLOATINGPOINT_EQ(solution[2], static_cast<SReal>(3))
    EXPECT_FLOATINGPOINT_EQ(solution[1], static_cast<SReal>(-21))
    EXPECT_FLOATINGPOINT_EQ(solution[0], static_cast<SReal>(38))
}

TEST(TriangularSystemSolver, computeLowerTriangularMatrixCoordinates)
{
    for (sofa::Index matrixSize = 2; matrixSize < 50; ++matrixSize)
    {
        const auto nbElementsInATriangularMatrix = matrixSize * (matrixSize+1) / 2;

        sofa::Index nbCoordinates {};
        for (sofa::Index row = 0; row < matrixSize; ++row)
        {
            for (sofa::Index col = 0; col <= row; ++col)
            {
                sofa::Index r, c;
                linearalgebra::computeRowColumnCoordinateFromIndexInLowerTriangularMatrix(nbCoordinates, r, c);
                EXPECT_EQ(r, row) << "index " << nbCoordinates << ", matrix size " << matrixSize;
                EXPECT_EQ(c, col) << "index " << nbCoordinates << ", matrix size " << matrixSize;
                ++nbCoordinates;
            }
        }

        EXPECT_EQ(nbCoordinates, nbElementsInATriangularMatrix);
    }
}

}
