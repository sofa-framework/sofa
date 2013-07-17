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

inline int NewProximityIntersection::doIntersectionPointPoint(SReal dist2, const Vector3& p, const Vector3& q, OutputVector* contacts, int id)
{
    Vector3 pq = q-p;

    SReal norm2 = pq.norm2();

    if ( norm2 >= dist2)
        return 0;

    //const SReal contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->value = helper::rsqrt(norm2);
    detection->normal=pq/detection->value;
    //detection->value -= contactDist;
    return 1;
}

template <class DataTypes1,class DataTypes2>
bool NewProximityIntersection::testIntersection(TSphere<DataTypes1>& e1, TSphere<DataTypes2>& e2)
{
    OutputVector contacts;
    const SReal alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    return doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), &contacts, -1) > 0;
}

template <class DataTypes>
bool NewProximityIntersection::testIntersection(Capsule&, TSphere<DataTypes>&){
    //you can do but not useful because it is not called
    return false;
}

template <class DataTypes>
int NewProximityIntersection::computeIntersection(Capsule & cap, TSphere<DataTypes> & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance()+cap.getProximity()+sph.getProximity(),getContactDistance()+cap.getProximity()+sph.getProximity(),contacts);
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
    return BaseIntTool::computeIntersection(sph1,sph2,sph1.getProximity() + sph2.getProximity() + getAlarmDistance(),sph1.getProximity() + sph2.getProximity() +getContactDistance(),contacts);
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
