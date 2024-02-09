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
#include <sofa/component/linearsystem/MatrixMapping.h>

namespace sofa::component::linearsystem
{

template <class TMatrix>
MatrixMapping<TMatrix>::~MatrixMapping() = default;

template <class TMatrix>
MatrixMapping<TMatrix>::MatrixMapping(const PairMechanicalStates& states)
{
    setPairStates(states);
}

template <class TMatrix>
bool MatrixMapping<TMatrix>::hasPairStates(
    const PairMechanicalStates& pairStates) const
{
    return l_mechanicalStates.size() >= 2 &&
        l_mechanicalStates[0] == pairStates[0] &&
        l_mechanicalStates[1] == pairStates[1];
}

template <class TMatrix>
void MatrixMapping<TMatrix>::setPairStates(const PairMechanicalStates& pairStates)
{
    l_mechanicalStates.clear();
    l_mechanicalStates.add(pairStates[0]);
    l_mechanicalStates.add(pairStates[1]);
}

}
