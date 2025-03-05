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
#include <sofa/component/linearsystem/config.h>

#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>
#include <sofa/component/linearsolver/iterative/LinearSystemData[GraphScattered].h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>

namespace sofa::component::linearsystem
{

using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;

template<>
void TypedMatrixLinearSystem<GraphScatteredMatrix, GraphScatteredVector>::setRHS(core::MultiVecDerivId v);

template<>
void TypedMatrixLinearSystem<GraphScatteredMatrix, GraphScatteredVector>::setSystemSolution(core::MultiVecDerivId v);

template<>
void TypedMatrixLinearSystem<GraphScatteredMatrix, GraphScatteredVector>::copyLocalVectorToGlobalVector(core::MultiVecDerivId v, GraphScatteredVector* globalVector);

template<>
void TypedMatrixLinearSystem<GraphScatteredMatrix, GraphScatteredVector>::dispatchSystemSolution(core::MultiVecDerivId v);

template<>
void TypedMatrixLinearSystem<GraphScatteredMatrix, GraphScatteredVector>::dispatchSystemRHS(core::MultiVecDerivId v);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSYSTEM_GRAPHSCATTERED_CPP)
    extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API TypedMatrixLinearSystem< GraphScatteredMatrix, GraphScatteredVector >;
#endif



}// namespace sofa::component::linearsystem
