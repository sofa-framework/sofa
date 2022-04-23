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
#define SOFA_COMPONENT_COLLISION_CAPSULEINTTOOL_CPP
#include <CollisionOBBCapsule/detection/intersection/CapsuleIntTool.inl>

namespace collisionobbcapsule::detection::intersection
{
using namespace sofa::defaulttype;
using namespace sofa::core::collision;

bool CapsuleIntTool::shareSameVertex(const geometry::Capsule & c1,const geometry::Capsule & c2){
    return c1.shareSameVertex(c2);
}

template COLLISIONOBBCAPSULE_API int CapsuleIntTool::computeIntersection(geometry::TCapsule<Vec3Types>&, geometry::TCapsule<Vec3Types>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template COLLISIONOBBCAPSULE_API int CapsuleIntTool::computeIntersection(geometry::TCapsule<Vec3Types>&, geometry::TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template COLLISIONOBBCAPSULE_API int CapsuleIntTool::computeIntersection(geometry::TCapsule<RigidTypes>&, geometry::TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template COLLISIONOBBCAPSULE_API int CapsuleIntTool::computeIntersection(geometry::TCapsule<RigidTypes> & cap, geometry::OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template COLLISIONOBBCAPSULE_API int CapsuleIntTool::computeIntersection(geometry::TCapsule<Vec3Types> & cap, geometry::OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);

class COLLISIONOBBCAPSULE_API CapsuleIntTool;
} // namespace collisionobbcapsule::detection::intersection
