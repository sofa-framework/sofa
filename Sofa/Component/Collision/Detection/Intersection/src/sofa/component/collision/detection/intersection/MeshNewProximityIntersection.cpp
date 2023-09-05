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
#include <sofa/component/collision/detection/intersection/MeshNewProximityIntersection.inl>

#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/collision/IntersectorFactory.h>


namespace sofa::component::collision::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

IntersectorCreator<NewProximityIntersection, MeshNewProximityIntersection> MeshNewProximityIntersectors("Mesh");

MeshNewProximityIntersection::MeshNewProximityIntersection(NewProximityIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.add<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);

        intersection->intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel, MeshNewProximityIntersection>(this);
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, MeshNewProximityIntersection>(this);

    }
}

bool MeshNewProximityIntersection::testIntersection(Point& pt1, Point& pt2)
{
    SOFA_UNUSED(pt1);
    SOFA_UNUSED(pt2);
    return false;
}

bool MeshNewProximityIntersection::testIntersection(Line& line, Point& pt)
{
    SOFA_UNUSED(line);
    SOFA_UNUSED(pt);
    return false;
}

bool MeshNewProximityIntersection::testIntersection(Line& line1, Line& line2)
{
    SOFA_UNUSED(line1);
    SOFA_UNUSED(line2);
    return false;
}

bool MeshNewProximityIntersection::testIntersection(Triangle& tri, Point& pt)
{
    SOFA_UNUSED(tri);
    SOFA_UNUSED(pt);
    return false;
}

bool MeshNewProximityIntersection::testIntersection(Triangle& tri, Line& line)
{
    SOFA_UNUSED(tri);
    SOFA_UNUSED(line);
    return false;
}

bool MeshNewProximityIntersection::testIntersection(Triangle& tri1, Triangle& tri2)
{
    SOFA_UNUSED(tri1);
    SOFA_UNUSED(tri2);
    return false;
}

int MeshNewProximityIntersection::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const int n = NewProximityIntersection::doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }

    return n;
}


int MeshNewProximityIntersection::computeIntersection(Line& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.p(), contacts, e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


int MeshNewProximityIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    const Index id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    const int n = doIntersectionLineLine(dist2, e1.p1(),e1.p2(), e2.p1(),e2.p2(), contacts, id);
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    const int n = doIntersectionTrianglePoint(dist2, e1.flags(),e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.p(), contacts, e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts)
{
    static_assert(std::is_same_v<Triangle::Coord, Line::Coord>, "Data mismatch");
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal    dist2 = alarmDist*alarmDist;
    const Triangle::Coord& p1 = e1.p1();
    const Triangle::Coord& p2 = e1.p2();
    const Triangle::Coord& p3 = e1.p3();
    const Triangle::Deriv& pn = e1.n();
    const Line::Coord& q1 = e2.p1();
    const Line::Coord& q2 = e2.p2();

    const int f1 = e1.flags();

    int n = 0;

    if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p1, contacts, e2.getIndex(), true);
    }
    if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p2, contacts, e2.getIndex(), true);
    }
    if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p3, contacts, e2.getIndex(), true);
    }

    n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, e2.getIndex(), false);
    n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, contacts, e2.getIndex(), false);

    if (intersection->useLineLine.getValue())
    {
        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
            n += doIntersectionLineLine(dist2, p1, p2, q1, q2, contacts, e2.getIndex());
        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
            n += doIntersectionLineLine(dist2, p2, p3, q1, q2, contacts, e2.getIndex());
        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
            n += doIntersectionLineLine(dist2, p3, p1, q1, q2, contacts, e2.getIndex());
    }

    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }

    return n;
}


int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Triangle& e2, OutputVector* contacts)
{
    if (e1.getIndex() >= e1.getCollisionModel()->getSize())
    {
        msg_error(intersection) << "computeIntersection(Triangle, Triangle): ERROR invalid e1 index "
            << e1.getIndex() << " on CM " << e1.getCollisionModel()->getName() << " of size " << e1.getCollisionModel()->getSize();
        return 0;
    }

    if (e2.getIndex() >= e2.getCollisionModel()->getSize())
    {
        msg_error(intersection) << "computeIntersection(Triangle, Triangle): ERROR invalid e2 index "
            << e2.getIndex() << " on CM " << e2.getCollisionModel()->getName() << " of size " << e2.getCollisionModel()->getSize();
        return 0;
    }

    const bool neighbor =  e1.getCollisionModel() == e2.getCollisionModel() &&
        (e1.p1Index()==e2.p1Index() || e1.p1Index()==e2.p2Index() || e1.p1Index()==e2.p3Index() ||
         e1.p2Index()==e2.p1Index() || e1.p2Index()==e2.p2Index() || e1.p2Index()==e2.p3Index() ||
         e1.p3Index()==e2.p1Index() || e1.p3Index()==e2.p2Index() || e1.p3Index()==e2.p3Index());


    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const SReal dist2 = alarmDist*alarmDist;
    const auto& p1 = e1.p1();
    const auto& p2 = e1.p2();
    const auto& p3 = e1.p3();
    auto& pn = e1.n();
    const auto& q1 = e2.p1();
    const auto& q2 = e2.p2();
    const auto& q3 = e2.p3();
    auto& qn = e2.n();


    if(neighbor)
        return 0;

    const int f1 = e1.flags();
    const int f2 = e2.flags();

    const Index id1 = e1.getIndex()*3; // index of contacts involving points in e1
    const Index id2 = e1.getCollisionModel()->getSize()*3 + e2.getIndex()*12; // index of contacts involving points in e2

    bool useNormal = true;
    const bool bothSide1 = e1.getCollisionModel()->d_bothSide.getValue();
    const bool bothSide2 = e2.getCollisionModel()->d_bothSide.getValue();


    if(bothSide1 && bothSide2)
        useNormal=false;
    else
        if(!bothSide1)
            qn = -pn;
        else
            if(!bothSide2)
                pn = -qn;

    int n = 0;
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p1, contacts, id1+0, true, useNormal);
    if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p2, contacts, id1+1, true, useNormal);
    if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p3, contacts, id1+2, true, useNormal);

    if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, false, useNormal);
    if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, contacts, id2+1, false, useNormal);
    if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q3, contacts, id2+2, false, useNormal);

    if (intersection->useLineLine.getValue())
    {
        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
        {
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
                n += doIntersectionLineLine(dist2, p1, p2, q1, q2, contacts, id2+3, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
                n += doIntersectionLineLine(dist2, p1, p2, q2, q3, contacts, id2+4, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
                n += doIntersectionLineLine(dist2, p1, p2, q3, q1, contacts, id2+5, pn, useNormal);
        }

        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
        {
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
                n += doIntersectionLineLine(dist2, p2, p3, q1, q2, contacts, id2+6, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
                n += doIntersectionLineLine(dist2, p2, p3, q2, q3, contacts, id2+7, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
                n += doIntersectionLineLine(dist2, p2, p3, q3, q1, contacts, id2+8, pn, useNormal);
        }

        if (f1&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
        {
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)
                n += doIntersectionLineLine(dist2, p3, p1, q1, q2, contacts, id2+9, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)
                n += doIntersectionLineLine(dist2, p3, p1, q2, q3, contacts, id2+10, pn, useNormal);
            if (f2&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)
                n += doIntersectionLineLine(dist2, p3, p1, q3, q1, contacts, id2+11, pn, useNormal);
        }
    }

    if (n>0)
    {
        const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();
        for (int i = 0; i < n; ++i)
        {
            (*contacts)[contacts->size()-n+i].elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            (*contacts)[contacts->size()-n+i].value -= contactDist;
        }
    }

    return n;
}



} // namespace sofa::component::collision::detection::intersection
