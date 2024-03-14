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
#include <sofa/core/AccumulationVecId.h>
#include <sofa/core/State.h>
#include <numeric>

namespace sofa::core
{

template <class TDataTypes, VecType vtype, VecAccess vaccess>
typename TDataTypes::Deriv
AccumulationVecId<TDataTypes, vtype, vaccess>::operator[](Size i) const
{
    return std::accumulate(m_contributingVecIds.begin(), m_contributingVecIds.end(), Deriv{},
       [i, this](Deriv a, core::ConstVecDerivId v)
       {
           return std::move(a) + m_state.read(v)->getValue()[i];
       });
}

template <class TDataTypes, VecType vtype, VecAccess vaccess>
void AccumulationVecId<TDataTypes, vtype, vaccess>::addToContributingVecIds(core::ConstVecDerivId vecDerivId)
{
    if (std::find(m_contributingVecIds.begin(), m_contributingVecIds.end(), vecDerivId) == m_contributingVecIds.end())
    {
        m_contributingVecIds.emplace_back(vecDerivId);
    }
}

template <class TDataTypes, VecType vtype, VecAccess vaccess>
void AccumulationVecId<TDataTypes, vtype, vaccess>::removeFromContributingVecIds(
    core::ConstVecDerivId vecDerivId)
{
    m_contributingVecIds.erase(
        std::remove(m_contributingVecIds.begin(), m_contributingVecIds.end(), vecDerivId)
        , m_contributingVecIds.end());
}


}
