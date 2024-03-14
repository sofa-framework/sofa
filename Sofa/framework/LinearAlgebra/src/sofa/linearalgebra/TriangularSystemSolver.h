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
#include <sofa/config.h>
#include <cstring>
#include <cmath>

namespace sofa::linearalgebra
{

/// Solves a lower unitriangular system where the matrix is represented in CSR format
///
/// Forward substitution is used to solve a linear system L*x = b, where:
/// L is a lower unitriangular matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The lower unitriangular matrix must be provided in compressed sparse row (CSR) format
///
/// \param systemSize The size of the system. All other parameters must comply with this size
/// \param rightHandSideVector The right-hand side vector
/// \param solutionVector The solution vector
/// \param CSR_rows The array storing the starting index of each row in the data array.
/// \param CSR_columns The array storing the column indices of the nonzero values in the data array.
/// \param CSR_values The array containing the nonzero values of the matrix
template<typename Real, typename Integer>
void solveLowerUnitriangularSystemCSR(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solutionVector,
    const Integer* const CSR_rows,
    const Integer* const CSR_columns,
    const Real* const CSR_values
    )
{
    for (sofa::Size i = 0; i < systemSize; ++i)
    {
        Real x_i = rightHandSideVector[i];
        for (Integer p = CSR_rows[i]; p < CSR_rows[i + 1]; ++p)
        {
            x_i -= CSR_values[p] * solutionVector[CSR_columns[p]];
        }
        solutionVector[i] = x_i;
    }
}

/// Solves a upper unitriangular system where the matrix is represented in CSR format
///
/// Backward substitution is used to solve a linear system U*x = b, where:
/// U is a upper unitriangular matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The upper unitriangular matrix must be provided in compressed sparse row (CSR) format
///
/// \param systemSize The size of the system. All other parameters must comply with this size
/// \param rightHandSideVector The right-hand side vector
/// \param solutionVector The solution vector
/// \param CSR_rows The array storing the starting index of each row in the data array.
/// \param CSR_columns The array storing the column indices of the nonzero values in the data array.
/// \param CSR_values The array containing the nonzero values of the matrix
template<typename Real, typename Integer>
void solveUpperUnitriangularSystemCSR(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solutionVector,
    const Integer* const CSR_rows,
    const Integer* const CSR_columns,
    const Real* const CSR_values
    )
{
    for (sofa::Size i = systemSize - 1; i != static_cast<sofa::Size>(-1); --i)
    {
        Real x_i = rightHandSideVector[i];
        for (Integer p = CSR_rows[i]; p < CSR_rows[i + 1]; ++p)
        {
            x_i -= CSR_values[p] * solutionVector[CSR_columns[p]];
        }
        solutionVector[i] = x_i;
    }
}

/// A lower triangular matrix can be stored as a linear array. This function
/// converts the index in this linear array to 2d coordinates (row and column)
/// of an element in the matrix.
///
/// Example of a 6x6 lower triangular matrix:
/// [ 0               ]
/// [ 1  2            ]
/// [ 3  4  5         ]
/// [ 6  7  8  9      ]
/// [10 11 12 13 14   ]
/// [15 16 17 18 19 20]
///
/// 0 => (0, 0)
/// 7 => (3, 1)
/// 18 => (5, 3)
inline void computeRowColumnCoordinateFromIndexInLowerTriangularMatrix(
    const sofa::Index flatIndex,
    sofa::Index& row,
    sofa::Index& col)
{
    row = std::floor(-0.5 + sqrt(0.25 + 2 * flatIndex));
    col = flatIndex - row * (row + 1) / 2;
}

}
