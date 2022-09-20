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

#include <sofa/component/collision/detection/algorithm/RayTraceDetection.h>

#include <sofa/core/ObjectFactory.h>
#include <numeric>

namespace sofa::component::collision::detection::algorithm
{

int RayTraceDetectionClass = core::RegisterObject(
        "Collision detection using TriangleOctreeModel").add<RayTraceDetection>();


void RayTraceDetection::doBaseObjectInit()
{
    const std::string broadPhaseComponentsString = sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::collision::BroadPhaseDetection>();
    const std::string narrowPhaseComponentsString = sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::collision::NarrowPhaseDetection>();

    msg_deprecated() << "As a replacement, use a BroadPhase component, such as [" << broadPhaseComponentsString
                     << "]," << msgendl
                     << "  AND a NarrowPhase component, such as [" << narrowPhaseComponentsString << "]." << msgendl
                     << "  " << BruteForceBroadPhase::GetClass()->className << " and " << RayTraceNarrowPhase::GetClass()->className << " have been automatically added to your scene for backward compatibility.";
}

} // namespace sofa::component::collision::detection::algorithm
