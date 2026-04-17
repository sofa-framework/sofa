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
#include <sofa/simulation/config.h>

#include <sofa/simulation/MappingGraph2.h>

namespace sofa::simulation
{
using MappingGraph = MappingGraph2;


template<class JacobianMatrixType>
class MappingJacobians
{
    const core::behavior::BaseMechanicalState& m_mappedState;

    std::map< core::behavior::BaseMechanicalState*, std::shared_ptr<JacobianMatrixType> > m_map;

public:

    MappingJacobians() = delete;
    MappingJacobians(const core::behavior::BaseMechanicalState& mappedState) : m_mappedState(mappedState) {}

    void addJacobianToTopMostParent(std::shared_ptr<JacobianMatrixType> jacobian, core::behavior::BaseMechanicalState* topMostParent)
    {
        m_map[topMostParent] = jacobian;
    }

    std::shared_ptr<JacobianMatrixType> getJacobianFrom(core::behavior::BaseMechanicalState* mstate) const
    {
        const auto it = m_map.find(mstate);
        if (it != m_map.end())
            return it->second;
        return nullptr;
    }
};

}
