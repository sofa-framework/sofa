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

/// Solves a linear system D*x = b, where:
/// D is a diagonal matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The diagonal matrix is stored as the list of entries in the diagonal
template<typename Real>
void solveDiagonalSystem(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solution,
    const Real* const D_values)
{
    for (sofa::Size i = 0 ; i < systemSize; ++i)
    {
        solution[i] = rightHandSideVector[i] / D_values[i];
    }
}

/// Solves a linear system D*x = b, where:
/// D is a diagonal matrix
/// x is the solution vector
/// b is the right-hand side vector
/// The diagonal matrix is stored as the list of the inverse of the entries in the diagonal
template<typename Real>
void solveDiagonalSystemUsingInvertedValues(
    const sofa::Size systemSize,
    const Real* rightHandSideVector,
    Real* solution,
    const Real* const Dinv_values)
{
    for (sofa::Size i = 0 ; i < systemSize; ++i)
    {
        solution[i] = rightHandSideVector[i] * Dinv_values[i];
    }
}

}
