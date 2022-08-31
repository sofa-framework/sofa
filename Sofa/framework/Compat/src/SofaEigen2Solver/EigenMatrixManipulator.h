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

#include <sofa/linearalgebra/EigenMatrixManipulator.h>

SOFA_DISABLED_HEADER("v21.12", "v22.06", "sofa/linearalgebra/EigenMatrixManipulator.h")
#include <SofaEigen2Solver/EigenBaseSparseMatrix.h>

namespace sofa::component::linearsolver
{

    typedef Eigen::SparseMatrix<SReal>    SparseMatrixEigen;
    typedef Eigen::SparseVector<SReal>    SparseVectorEigen;
    typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1>       VectorEigen;

    using LLineManipulator = sofa::linearalgebra::LLineManipulator;
    using LMatrixManipulator = sofa::linearalgebra::LMatrixManipulator;

} // namespace sofa::component::linearsolver
