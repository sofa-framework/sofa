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

template <class DataTypes1,class DataTypes2>
bool NewProximityIntersection::testIntersection(TSphere<DataTypes1>& e1, TSphere<DataTypes2>& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), &contacts, -1);
    return n>0;
}

template <class DataTypes>
bool NewProximityIntersection::testIntersection(Capsule&, TSphere<DataTypes>&){
    //you can do but not useful because it is not called
    return false;
}

template <class DataTypes>
int NewProximityIntersection::computeIntersection(Capsule & cap, TSphere<DataTypes> & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance(),getContactDistance(),contacts);
}

template <class DataTypes>
int NewProximityIntersection::computeIntersection(TSphere<DataTypes> & sph, OBB & box,OutputVector* contacts){
    return OBBIntTool::computeIntersection(sph,box,sph.getProximity() + box.getProximity() + getAlarmDistance(),box.getProximity() + sph.getProximity() + getContactDistance(),contacts);
}

template <class DataTypes>
bool NewProximityIntersection::testIntersection(TSphere<DataTypes> &,OBB &){
    return true;
}

template <class DataTypes1,class DataTypes2>
int NewProximityIntersection::computeIntersection(TSphere<DataTypes1>& sph1, TSphere<DataTypes2>& sph2, OutputVector* contacts)
{
//    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
//    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
//    if (n>0)
//    {
//        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
//        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
//        {
//            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
//            detection->value -= contactDist;
//        }
//    }
//    return n;
    double r = sph1.r() + sph2.r();
    double alarmDist = getAlarmDistance() + r;
    Vector3 dist = sph2.center() - sph1.center();

    if (dist.norm2() >= alarmDist*alarmDist)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->normal = dist;
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->value = distSph1Sph2 - r - getContactDistance() - sph1.getProximity() - sph2.getProximity();
    detection->elem.first = sph1;
    detection->elem.second = sph2;
    detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

    return 1;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
