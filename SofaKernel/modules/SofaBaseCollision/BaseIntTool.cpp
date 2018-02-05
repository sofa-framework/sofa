/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseCollision/BaseIntTool.h>

namespace sofa{namespace component{namespace collision{

//template<>
bool BaseIntTool::testIntersection(Cube &cube1, Cube &cube2,SReal alarmDist)
{
    const defaulttype::Vector3& minVect1 = cube1.minVect();
    const defaulttype::Vector3& minVect2 = cube2.minVect();
    const defaulttype::Vector3& maxVect1 = cube1.maxVect();
    const defaulttype::Vector3& maxVect2 = cube2.maxVect();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

class SOFA_BASE_COLLISION_API BaseIntTool;

}}}
