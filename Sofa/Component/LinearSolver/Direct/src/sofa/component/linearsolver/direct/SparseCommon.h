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

#include <sofa/component/linearsolver/direct/config.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>

namespace sofa::component::linearsolver::direct
{
/** 
compute the adjency matrix in CSR format from the matrix given in CSR format, we assume that the given matrix is symmetric

M_colptr[i+1]-M_colptr[i] is the number of non null values on the i-th line of the matrix
M_rowind[M_colptr[i]] to M_rowind[M_colptr[i+1]] is the list of the indices of the columns containing a non null value on the i-th line

xadj[i+1]-xadj[i] is the number of neighbors of the i-th node
adj[xadj[i]] is the first neighbor of the i-th node

**/

// compare the shape of two matrix given in CSR format, return false if the matrices have the same shape and return true if their shapes are different
inline bool compareMatrixShape(int s_M, int * M_colptr,int * M_rowind, int s_P, int * P_colptr,int * P_rowind) {
    if (s_M != s_P) return true;
    if (M_colptr[s_M] != P_colptr[s_M] ) return true;

    for (int i=0;i<s_P;i++) {
        if (M_colptr[i]!=P_colptr[i]) return true;
    }

    for (int i=0;i<M_colptr[s_M];i++) {
        if (M_rowind[i]!=P_rowind[i]) return true;
    }

    return false;
}

} // namespace sofa::component::linearsolver::direct

