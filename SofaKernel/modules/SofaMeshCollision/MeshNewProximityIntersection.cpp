/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMeshCollision/MeshNewProximityIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

IntersectorCreator<NewProximityIntersection, MeshNewProximityIntersection> MeshNewProximityIntersectors("Mesh");

MeshNewProximityIntersection::MeshNewProximityIntersection(NewProximityIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.add<PointModel, PointModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<SphereModel, PointModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineModel, PointModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineModel, SphereModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineModel, LineModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, PointModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, SphereModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, LineModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, TriangleModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<CapsuleModel, TriangleModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<CapsuleModel, LineModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, OBBModel, MeshNewProximityIntersection>(this);

        intersection->intersectors.add<RigidSphereModel, PointModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineModel, RigidSphereModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleModel, RigidSphereModel, MeshNewProximityIntersection>(this);

        intersection->intersectors.add<RigidCapsuleModel, TriangleModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<RigidCapsuleModel, LineModel, MeshNewProximityIntersection>(this);
    }
}

int MeshNewProximityIntersection::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    DetectionOutput detection;
    int n = intersection->doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), detection, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
    }

    return n;
}


int MeshNewProximityIntersection::computeIntersection(Line& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    DetectionOutput detection;
    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.p(), detection, e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
    }
    return n;
}


int MeshNewProximityIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    const int id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    DetectionOutput detection;
    int n = doIntersectionLineLine(dist2, e1.p1(),e1.p2(), e2.p1(),e2.p2(), detection, id);
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
    }
    return n;
}

int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    DetectionOutput detection;
    int n = doIntersectionTrianglePoint(dist2, e1.flags(),e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.p(), detection, e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
    }
    return n;
}


int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal    dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();

    const int f1 = e1.flags();

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
    int n = 0;
    DetectionOutput detection;

    if (f1&TriangleModel::FLAG_P1)
    {
        if (doIntersectionLinePoint(dist2, q1, q2, p1, detection, e2.getIndex(), true))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }        
    }
    if (f1&TriangleModel::FLAG_P2)
    {
        if (doIntersectionLinePoint(dist2, q1, q2, p2, detection, e2.getIndex(), true))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (f1&TriangleModel::FLAG_P3)
    {
        if ( doIntersectionLinePoint(dist2, q1, q2, p3, detection, e2.getIndex(), true) )
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }

    if( doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, detection, e2.getIndex(), false) )
    {
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
        ++n;
    }

    if( doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, detection, e2.getIndex(), false) )
    {
        detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
        detection.value -= contactDist;
        contacts->addContact(&detection);
        ++n;
    }

    if (intersection->useLineLine.getValue())
    {
        if (f1&TriangleModel::FLAG_E12)
        {
            if ( doIntersectionLineLine(dist2, p1, p2, q1, q2, detection, e2.getIndex()) )
            {
                detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                detection.value -= contactDist;
                contacts->addContact(&detection);
                ++n;
            }
        }
        if (f1&TriangleModel::FLAG_E23)
        {
            if (doIntersectionLineLine(dist2, p2, p3, q1, q2, detection, e2.getIndex()) )
            {
                detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                detection.value -= contactDist;
                contacts->addContact(&detection);
                ++n;
            }
        }
        if (f1&TriangleModel::FLAG_E31)
        {
            if (doIntersectionLineLine(dist2, p3, p1, q1, q2, detection, e2.getIndex()) )
            {
                detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                detection.value -= contactDist;
                contacts->addContact(&detection);
                ++n;
            }
        }
    }
    return n;
}


int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Triangle& e2, OutputVector* contacts)
{
    if (e1.getIndex() >= e1.getCollisionModel()->getSize())
    {
        intersection->serr << "NewProximityIntersection::computeIntersection(Triangle, Triangle): ERROR invalid e1 index "
                << e1.getIndex() << " on CM " << e1.getCollisionModel()->getName() << " of size " << e1.getCollisionModel()->getSize()<<intersection->sendl;
        return 0;
    }

    if (e2.getIndex() >= e2.getCollisionModel()->getSize())
    {
        intersection->serr << "NewProximityIntersection::computeIntersection(Triangle, Triangle): ERROR invalid e2 index "
                << e2.getIndex() << " on CM " << e2.getCollisionModel()->getName() << " of size " << e2.getCollisionModel()->getSize()<<intersection->sendl;
        return 0;
    }

    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();
    const Vector3& q3 = e2.p3();
    const Vector3& qn = e2.n();

    const int f1 = e1.flags();
    const int f2 = e2.flags();

    const int id1 = e1.getIndex()*3; // index of contacts involving points in e1
    const int id2 = e1.getCollisionModel()->getSize()*3 + e2.getIndex()*12; // index of contacts involving points in e2

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
    int n = 0;
    DetectionOutput detection;

    if (f1&TriangleModel::FLAG_P1)
    {
        if ( doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p1, detection, id1 + 0, true))
        {           
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
        
    if (f1&TriangleModel::FLAG_P2)
    {
        if (doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p2, detection, id1 + 1, true))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (f1&TriangleModel::FLAG_P3)
    {
        if (doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p3, detection, id1 + 2, true))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (f2&TriangleModel::FLAG_P1)
    {
        if (doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, detection, id2 + 0, false) )
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (f2&TriangleModel::FLAG_P2)
    {
        if (doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, detection, id2 + 1, false))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (f2&TriangleModel::FLAG_P3)
    {
        if (doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q3, detection, id2 + 2, false))
        {
            detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection.value -= contactDist;
            contacts->addContact(&detection);
            ++n;
        }
    }
    if (intersection->useLineLine.getValue())
    {
        if (f1&TriangleModel::FLAG_E12)
        {
            if (f2&TriangleModel::FLAG_E12)
            {
                if (doIntersectionLineLine(dist2, p1, p2, q1, q2, detection, id2 + 3))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
            if (f2&TriangleModel::FLAG_E23)
            {
                if (doIntersectionLineLine(dist2, p1, p2, q2, q3, detection, id2 + 4))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                if (doIntersectionLineLine(dist2, p1, p2, q3, q1, detection, id2 + 5))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
        }

        if (f1&TriangleModel::FLAG_E23)
        {
            if (f2&TriangleModel::FLAG_E12)
            {
                if (doIntersectionLineLine(dist2, p2, p3, q1, q2, detection, id2 + 6))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }            
            if (f2&TriangleModel::FLAG_E23)
            {
                if (doIntersectionLineLine(dist2, p2, p3, q2, q3, detection, id2 + 7))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                if (doIntersectionLineLine(dist2, p2, p3, q3, q1, detection, id2 + 8))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
        }

        if (f1&TriangleModel::FLAG_E31)
        {
            if (f2&TriangleModel::FLAG_E12)
            {
                if (doIntersectionLineLine(dist2, p3, p1, q1, q2, detection, id2 + 9))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
            if (f2&TriangleModel::FLAG_E23)
            {
                if (doIntersectionLineLine(dist2, p3, p1, q2, q3, detection, id2 + 10))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                if (doIntersectionLineLine(dist2, p3, p1, q3, q1, detection, id2 + 11))
                {
                    detection.elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
                    detection.value -= contactDist;
                    contacts->addContact(&detection);
                    ++n;
                }
            }
        }
    }
    return n;
}



} // namespace collision

} // namespace component

} // namespace sofa

