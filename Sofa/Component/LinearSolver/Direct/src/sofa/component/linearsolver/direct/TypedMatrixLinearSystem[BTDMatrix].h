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
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/linearalgebra/BTDMatrix.h>
#include <sofa/linearalgebra/BlockVector.h>

#if !defined(SOFA_COMPONENT_LINEARSOLVER_TYPEDMATRIXLINEARSYSTEM_BTDMATRIX_CPP)
namespace sofa::component::linearsystem
{
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API TypedMatrixLinearSystem< linearalgebra::BTDMatrix<6, SReal>, linearalgebra::BlockVector<6, SReal> >;
}
#endif
