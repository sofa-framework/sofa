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

namespace sofa::linearalgebra
{

/// Use forward substitution to solve a linear system L*x = b, where:
/// L is a lower triangular matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The lower triangular matrix must be provided in compressed sparse row (CSR) format
template<typename Real, typename Integer>
void solveLowerTriangularSystem(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solution,
    const Integer* const L_columns,
    const Integer* const L_row,
    const Real* const L_values
    )
{
    for (sofa::Size i = 0; i < systemSize; ++i)
    {
        Real x_i = rightHandSideVector[i];
        for (Integer p = L_columns[i]; p < L_columns[i + 1]; ++p)
        {
            x_i -= L_values[p] * solution[L_row[p]];
        }
        solution[i] = x_i;
    }
}

/// Use backward substitution to solve a linear system U*x = b, where:
/// U is a upper triangular matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The upper triangular matrix must be provided in compressed sparse row (CSR) format
template<typename Real, typename Integer>
void solveUpperTriangularSystem(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solution,
    const Integer* const U_columns,
    const Integer* const U_row,
    const Real* const U_values
    )
{
    for (sofa::Size i = systemSize - 1; i != static_cast<sofa::Size>(-1); --i)
    {
        Real x_i = rightHandSideVector[i];
        for (Integer p = U_columns[i]; p < U_columns[i + 1]; ++p)
        {
            x_i -= U_values[p] * solution[U_row[p]];
        }
        solution[i] = x_i;
    }
}

}
