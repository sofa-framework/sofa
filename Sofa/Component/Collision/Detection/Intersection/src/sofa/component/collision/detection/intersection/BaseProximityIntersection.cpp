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
#include <sofa/component/collision/detection/intersection/BaseProximityIntersection.h>

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::component::collision::geometry;

BaseProximityIntersection::BaseProximityIntersection()
    : d_alarmDistance(initData(&d_alarmDistance, 1.0_sreal, "alarmDistance", "Distance above which the intersection computations ignores the promixity pair. This distance can also be used in some broad phase algorithms to reduce the search area"))
    , d_contactDistance(initData(&d_contactDistance, 0.5_sreal, "contactDistance", "Distance below which a contact is created"))
{
    d_alarmDistance.setRequired(true);
    d_contactDistance.setRequired(true);

    alarmDistance.setOriginalData(&d_alarmDistance);
    contactDistance.setOriginalData(&d_contactDistance);
}


bool BaseProximityIntersection::testIntersection(Cube& cube1, Cube& cube2, const core::collision::Intersection* currentIntersection)
{
    const auto& minVect1 = cube1.minVect();
    const auto& minVect2 = cube2.minVect();
    const auto& maxVect1 = cube1.maxVect();
    const auto& maxVect2 = cube2.maxVect();

    const auto alarmDist = currentIntersection->getAlarmDistance() + cube1.getProximity() + cube2.getProximity();

    for (int i = 0; i < 3; i++)
    {
        if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
            return false;
    }

    return true;
}

int BaseProximityIntersection::computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(cube1);
    SOFA_UNUSED(cube2);
    SOFA_UNUSED(contacts);
    SOFA_UNUSED(currentIntersection);

    return 0;
}

} // namespace sofa::component::collision::detection::intersection
