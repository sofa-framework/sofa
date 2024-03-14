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
#include <sofa/core/BaseState.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core
{


bool BaseState::insertInNode( objectmodel::BaseNode* node )
{
    node->addState(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseState::removeInNode( objectmodel::BaseNode* node )
{
    node->removeState(this);
    Inherit1::removeInNode(node);
    return true;
}

void BaseState::addToTotalForces(core::ConstVecDerivId forceId)
{
    SOFA_UNUSED(forceId);
}

void BaseState::removeFromTotalForces(core::ConstVecDerivId forceId)
{
    SOFA_UNUSED(forceId);
}
} // namespace sofa::core

