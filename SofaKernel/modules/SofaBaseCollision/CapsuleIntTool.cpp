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
#define SOFA_COMPONENT_COLLISION_CAPSULEINTTOOL_CPP
#include <SofaBaseCollision/CapsuleIntTool.inl>

namespace sofa
{
namespace component
{
namespace collision
{
using namespace sofa::defaulttype;
using namespace sofa::core::collision;

bool CapsuleIntTool::shareSameVertex(const Capsule & c1,const Capsule & c2){
    return c1.shareSameVertex(c2);
}

template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<Vec3Types>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);

class SOFA_BASE_COLLISION_API CapsuleIntTool;
}
}
}
