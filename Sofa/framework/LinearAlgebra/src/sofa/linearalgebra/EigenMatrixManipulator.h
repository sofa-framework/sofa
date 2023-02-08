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
#include <sofa/linearalgebra/config.h>

#ifndef SOFA_BUILD_SOFA_LINEARALGEBRA
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v22.06", "v23.06")
#endif // SOFA_BUILD_SOFA_LINEARALGEBRA

#include <sofa/type/vector.h>

#include <Eigen/Core>
#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif
#include <Eigen/Sparse>

namespace sofa::linearalgebra
{

typedef Eigen::SparseMatrix<SReal>    SparseMatrixEigen;
typedef Eigen::SparseVector<SReal>    SparseVectorEigen;
typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1>       VectorEigen;

SOFA_LINEARALGEBRA_API
SOFA_MATRIXMANIPULATOR_DISABLED()
typedef DeprecatedAndRemoved LLineManipulator;

SOFA_LINEARALGEBRA_API
SOFA_MATRIXMANIPULATOR_DISABLED()
typedef DeprecatedAndRemoved LMatrixManipulator;

} // namespace sofa::linearalgebra
