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
#include <sofa/component/linearsystem/LinearSystemData.h>
#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>

namespace sofa::component::linearsystem
{
using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;

template<>
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::createSystemRHSVector();

template<>
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::createSystemSolutionVector();

template<>
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::resizeSystem(sofa::Size n);

template<>
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::clearSystem();

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_LINEARSYSTEMDATA_GRAPHSCATTERED_CPP)
extern template struct SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API LinearSystemData< GraphScatteredMatrix, GraphScatteredVector >;
#endif

}
