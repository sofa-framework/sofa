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
#include <SofaBaseLinearSolver/config.h>
#include <sofa/type/vector.h>
namespace sofa::component::linearsolver
{
template<typename T> class SOFA_SOFABASELINEARSOLVER_API FullMatrix;
template<typename T> class SOFA_SOFABASELINEARSOLVER_API LPtrFullMatrix;

template<typename T> class SOFA_SOFABASELINEARSOLVER_API SparseMatrix ;
template<typename T> class SOFA_SOFABASELINEARSOLVER_API DiagonalMatrix;

template<std::size_t LC, typename T = double>
class SOFA_SOFABASELINEARSOLVER_API BlockDiagonalMatrix;

template<typename T> class SOFA_SOFABASELINEARSOLVER_API FullMatrix;
template<typename T> class SOFA_SOFABASELINEARSOLVER_API FullVector;

template<typename TBloc, typename TVecBloc = type::vector<TBloc>, typename TVecIndex = type::vector<sofa::Index>>
class SOFA_SOFABASELINEARSOLVER_API CompressedRowSparseMatrix;

} // namespace sofa::component::linearsolver
