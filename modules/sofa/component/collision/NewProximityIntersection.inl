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
#include <sofa/component/collision/NewProximityIntersection.h>
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

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

inline int NewProximityIntersection::doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, OutputVector* contacts, int id)
{
    Vector3 pq;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    //detection->value -= contactDist;
    return 1;
}

template<class Sphere>
bool NewProximityIntersection::testIntersection(Sphere& e1, Sphere& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), &contacts, -1);
    return n>0;
}

template<class Sphere>
int NewProximityIntersection::computeIntersection(Sphere& e1, Sphere& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
