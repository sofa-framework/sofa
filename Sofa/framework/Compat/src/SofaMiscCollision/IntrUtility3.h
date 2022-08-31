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
#pragma once

#include <sofa/config.h>

#if __has_include(<CollisionOBBCapsule/detection/intersection/IntrUtility3.h>)
#include <CollisionOBBCapsule/detection/intersection/IntrUtility3.h>
#define COLLISIONOBBCAPSULE_INTRUTILITY3

SOFA_DEPRECATED_HEADER("v22.06", "v23.06", "CollisionOBBCapsule/detection/intersection/IntrUtility3.h")

#else
#error "OBB and Capsule-related contents has been moved to CollisionOBBCapsule. Include <CollisionOBBCapsule/detection/intersection/IntrUtility3.h> instead of this one."
#endif

#ifdef COLLISIONOBBCAPSULE_INTRUTILITY3

namespace sofa::component::collision
{
	template <typename TReal>
	using MyBox = collisionobbcapsule::detection::intersection::MyBox<TReal>;
	template <typename Real>
	using IntrConfiguration = collisionobbcapsule::detection::intersection::IntrConfiguration<Real>;
	template <typename Real>
	using CapIntrConfiguration = collisionobbcapsule::detection::intersection::CapIntrConfiguration<Real>;
	template <typename Real>
	using IntrUtil = collisionobbcapsule::detection::intersection::IntrUtil<Real>;
	template <class Primitive1Class,class Primitive2Class>
	using IntrAxis = collisionobbcapsule::detection::intersection::IntrAxis<Primitive1Class,Primitive2Class>;
	template <typename Real>
	using IntrConfigManager = collisionobbcapsule::detection::intersection::IntrConfigManager<Real>;
	template <class Primitive1Class,class Primitive2Class>
	using FindContactSet = collisionobbcapsule::detection::intersection::FindContactSet<Primitive1Class,Primitive2Class>;

	template <typename Real>
	void ClipConvexPolygonAgainstPlane (const type::Vec<3,Real>& normal,
    Real bonstant, int& quantity, type::Vec<3,Real>* P)
    {
    	collisionobbcapsule::detection::intersection::ClipConvexPolygonAgainstPlane(normal, bonstant, quantity, P);
    }

	template <typename TReal>
	type::Vec<3,TReal> GetPointFromIndex (int index, const MyBox<TReal>& box)
	{
    	return collisionobbcapsule::detection::intersection::GetPointFromIndex(index, box);
	}

	template <typename TDataTypes>
	type::Vec<3,typename TDataTypes::Real> getPointFromIndex (int index, const TOBB<TDataTypes>& box)
	{
    	return collisionobbcapsule::detection::intersection::getPointFromIndex(index, box);
	}


} // namespace sofa::component::collision

#endif // COLLISIONOBBCAPSULE_INTRUTILITY3
