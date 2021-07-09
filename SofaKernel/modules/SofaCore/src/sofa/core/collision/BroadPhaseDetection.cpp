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
#include <sofa/core/collision/BroadPhaseDetection.h>

namespace sofa::core::collision
{

void BroadPhaseDetection::beginBroadPhase()
{
    cmPairs.clear();
}

void BroadPhaseDetection::addCollisionModels(const sofa::type::vector<core::CollisionModel *>& v)
{
    for (auto* collisionModel : v)
    {
        addCollisionModel(collisionModel);
    }
}

void BroadPhaseDetection::endBroadPhase()
{
}

auto BroadPhaseDetection::getCollisionModelPairs() -> sofa::type::vector<CollisionModelPair>&
{
    return cmPairs;
}

auto BroadPhaseDetection::getCollisionModelPairs() const -> const sofa::type::vector<CollisionModelPair>&
{
    return cmPairs;
}

void BroadPhaseDetection::changeInstanceBP(Instance inst)
{
    storedCmPairs[instance].swap(cmPairs);
    cmPairs.swap(storedCmPairs[inst]);
}

} // namespace sofa::core::collision