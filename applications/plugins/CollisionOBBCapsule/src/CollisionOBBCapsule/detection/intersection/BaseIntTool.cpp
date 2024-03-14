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
#include <CollisionOBBCapsule/detection/intersection/BaseIntTool.h>

namespace collisionobbcapsule::detection::intersection
{

//template<>
bool BaseIntTool::testIntersection(sofa::component::collision::geometry::Cube &cube1, sofa::component::collision::geometry::Cube &cube2,SReal alarmDist)
{
    if (cube1 == cube2)
    {
        if (cube1.getConeAngle() < M_PI / 2)
            return false;
        else
            return true;
    }

    const type::Vec3& minVect1 = cube1.minVect();
    const type::Vec3& minVect2 = cube2.minVect();
    const type::Vec3& maxVect1 = cube1.maxVect();
    const type::Vec3& maxVect2 = cube2.maxVect();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

class COLLISIONOBBCAPSULE_API BaseIntTool;

} // namespace collisionobbcapsule::detection::intersection
