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
#define SOFA_COMPONENT_COLLISION_INTRUTILITY3_CPP
#include <CollisionOBBCapsule/detection/intersection/IntrUtility3.inl>

namespace collisionobbcapsule::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
template struct COLLISIONOBBCAPSULE_API IntrUtil<SReal>;

template struct COLLISIONOBBCAPSULE_API IntrUtil<geometry::TOBB<RigidTypes> >;
template class COLLISIONOBBCAPSULE_API IntrConfiguration<SReal>;
template struct COLLISIONOBBCAPSULE_API IntrConfigManager<SReal>;
template struct COLLISIONOBBCAPSULE_API IntrConfigManager<geometry::TOBB<Rigid3Types> >;
template class COLLISIONOBBCAPSULE_API IntrAxis<geometry::TOBB<Rigid3Types> >;
template class COLLISIONOBBCAPSULE_API FindContactSet<geometry::TOBB<Rigid3Types> >;
template COLLISIONOBBCAPSULE_API void ClipConvexPolygonAgainstPlane<SReal> (const Vec3&, SReal,int&, Vec3*);
template COLLISIONOBBCAPSULE_API Vec3 GetPointFromIndex<SReal> (int, const MyBox<SReal>&);
template COLLISIONOBBCAPSULE_API Vec<3,Rigid3Types::Real> getPointFromIndex<Rigid3Types> (int index, const geometry::TOBB<Rigid3Types>& box);
template class COLLISIONOBBCAPSULE_API CapIntrConfiguration<SReal>;

//----------------------------------------------------------------------------
} // namespace collisionobbcapsule::detection::intersection
