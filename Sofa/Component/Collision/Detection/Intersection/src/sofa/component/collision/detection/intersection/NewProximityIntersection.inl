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
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/Vec.h>


namespace sofa::component::collision::detection::intersection
{

inline int NewProximityIntersection::doIntersectionPointPoint(SReal dist2,
                                                       const type::Vec3& p,
                                                       const type::Vec3& q,
                                                       OutputVector* contacts,
                                                       int id)
{
    const type::Vec3 pq = q-p;

    const SReal norm2 = pq.norm2();

    if ( norm2 >= dist2)
        return 0;

    sofa::core::collision::DetectionOutput detectionOutput;
    detectionOutput.id = id;
    detectionOutput.point[0]=p;
    detectionOutput.point[1]=q;
    detectionOutput.value = helper::rsqrt(norm2);
    detectionOutput.normal = pq / detectionOutput.value;

    contacts->push_back(std::move(detectionOutput));

    return 1;
}

template<typename SphereType1, typename SphereType2>
bool NewProximityIntersection::testIntersection(SphereType1& sph1, SphereType2& sph2)
{
    const auto alarmDist = this->getAlarmDistance() + sph1.getProximity() + sph2.getProximity();

    OutputVector contacts;
    const double alarmDist2 = alarmDist + sph1.r() + sph2.r();
    const int n = doIntersectionPointPoint(alarmDist2 * alarmDist2, sph1.center(), sph2.center(), &contacts, -1);
    return n > 0;
}

template<typename SphereType1, typename SphereType2>
int NewProximityIntersection::computeIntersection(SphereType1& sph1, SphereType2& sph2, OutputVector* contacts)
{
    const auto alarmDist = this->getAlarmDistance() + sph1.getProximity() + sph2.getProximity();
    const auto contactDist = this->getContactDistance() + sph1.getProximity() + sph2.getProximity();

    const double alarmDist2 = alarmDist + sph1.r() + sph2.r();
    const int n = doIntersectionPointPoint(alarmDist2 * alarmDist2, sph1.center(), sph2.center(), contacts, (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex());
    if (n > 0)
    {
        const double contactDist2 = contactDist + sph1.r() + sph2.r();
        for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(sph1, sph2);
            detection->value -= contactDist2;
        }
    }
    return n;
}


} // namespace sofa::component::collision::detection::intersection
