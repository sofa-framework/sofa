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
#define SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_LINEARSYSTEMDATA_GRAPHSCATTERED_CPP
#include <sofa/component/linearsolver/iterative/LinearSystemData[GraphScattered].h>

namespace sofa::component::linearsystem
{

using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;


template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::createSystemRHSVector()
{
    rhs = std::make_unique<GraphScatteredVector>(nullptr, core::VecDerivId::null());
}

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::createSystemSolutionVector()
{
    solution = std::make_unique<GraphScatteredVector>(nullptr, core::VecDerivId::null());
}

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::resizeSystem(sofa::Size n)
{
    SOFA_UNUSED(n);
}

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void LinearSystemData<GraphScatteredMatrix, GraphScatteredVector>::clearSystem()
{
    allocateSystem();

    if (rhs)
    {
        rhs->reset();
    }

    if (solution)
    {
        solution->reset();
    }
}

template struct SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API LinearSystemData< GraphScatteredMatrix, GraphScatteredVector >;

} //namespace sofa::component::linearsystem

