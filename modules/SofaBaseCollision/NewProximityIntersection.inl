/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_INL

#include <sofa/helper/system/config.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{

inline int NewProximityIntersection::doIntersectionPointPoint(SReal dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id)
{
    defaulttype::Vector3 pq = q-p;

    SReal norm2 = pq.norm2();

    if ( norm2 >= dist2)
        return 0;

    //const SReal contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->value = helper::rsqrt(norm2);
    detection->normal=pq/detection->value;
    //detection->value -= contactDist;
    return 1;
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
