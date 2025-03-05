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
#ifndef SOFA_SCENEUTILS_H
#define SOFA_SCENEUTILS_H

#include <SceneCreator/config.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sofa/core/VecId.h>


namespace sofa::modeling
{

typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1> Vector;

typedef Eigen::Matrix<SReal, Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
typedef Eigen::SparseMatrix<SReal, Eigen::RowMajor> SparseMatrix;

/// Get a state vector from the scene graph. Includes only the independent state values, or also the
/// mapped ones, depending on the flag.
SOFA_SCENECREATOR_API Vector getVector( core::ConstVecId id, bool independentOnly=true );

}

#endif
