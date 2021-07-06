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
#include <SofaGeneralMeshCollision/DirectSAP.h>
#include <sofa/core/ObjectFactory.h>
#include <numeric>

namespace sofa::component::collision
{
    int DirectSAPClass = core::RegisterObject("Collision detection using sweep and prune")
        .add< DirectSAP >();


void DirectSAP::init()
{
    std::vector<std::string> broadPhaseComponents;
    std::vector<std::string> narrowPhaseComponents;
    findAllDetectionComponents(broadPhaseComponents, narrowPhaseComponents);

    if (broadPhaseComponents.empty())
    {
        broadPhaseComponents.push_back(BruteForceBroadPhase::GetClass()->className);
    }
    if (narrowPhaseComponents.empty())
    {
        narrowPhaseComponents.push_back(DirectSAPNarrowPhase::GetClass()->className);
    }

    const auto comma_fold = [](std::string a, std::string b)
    {
        return std::move(a) + ", " + std::move(b);
    };

    const std::string broadPhaseComponentsString = std::accumulate(
            std::next(broadPhaseComponents.begin()), broadPhaseComponents.end(),
            broadPhaseComponents[0],
            comma_fold);

    const std::string narrowPhaseComponentsString = std::accumulate(
            std::next(narrowPhaseComponents.begin()), narrowPhaseComponents.end(),
            narrowPhaseComponents[0],
            comma_fold);

    msg_deprecated() << "As a replacement, use a BroadPhase component such as [" << broadPhaseComponentsString << "] " << msgendl
                     << "  AND a NarrowPhase component such as [" << narrowPhaseComponentsString << "]." << msgendl
                     << "  " << BruteForceBroadPhase::GetClass()->className << " and " << DirectSAPNarrowPhase::GetClass()->className
                     << " have been automatically added to your scene for backward compatibility.";
}

void DirectSAP::findAllDetectionComponents(std::vector<std::string> &broadPhaseComponents,
                                                     std::vector<std::string> &narrowPhaseComponents)
{
    std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);

    for (const auto &entry : entries)
    {
        const auto creatorEntry = entry->creatorMap.begin();
        if (creatorEntry != entry->creatorMap.end())
        {
            const sofa::core::objectmodel::BaseClass *baseClass = creatorEntry->second->getClass();
            if (baseClass)
            {
                if (baseClass->hasParent(sofa::core::collision::BroadPhaseDetection::GetClass()))
                {
                    broadPhaseComponents.push_back(baseClass->className);
                }
                if (baseClass->hasParent(sofa::core::collision::NarrowPhaseDetection::GetClass()))
                {
                    narrowPhaseComponents.push_back(baseClass->className);
                }
            }
        }
    }
}
} // namespace sofa::component::collision


