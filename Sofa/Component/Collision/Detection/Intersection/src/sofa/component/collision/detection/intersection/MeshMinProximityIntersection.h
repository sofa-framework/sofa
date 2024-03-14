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

#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>

namespace sofa::component::collision::detection::intersection
{

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshMinProximityIntersection : public core::collision::BaseIntersector
{
    typedef MinProximityIntersection::OutputVector OutputVector;

public:
    MeshMinProximityIntersection(MinProximityIntersection* object, bool addSelf=true);

    bool testIntersection(collision::geometry::Point&, collision::geometry::Point&);
    template<class T> bool testIntersection(collision::geometry::TSphere<T>&, collision::geometry::Point&);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Point&);
    template<class T> bool testIntersection(collision::geometry::Line&, collision::geometry::TSphere<T>&);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Line&);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Point&);
    template<class T> bool testIntersection(collision::geometry::Triangle&, collision::geometry::TSphere<T>&);

    int computeIntersection(collision::geometry::Point&, collision::geometry::Point&, OutputVector*);
    template<class T> int computeIntersection(collision::geometry::TSphere<T>&, collision::geometry::Point&, OutputVector*);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Point&, OutputVector*);
    template<class T> int computeIntersection(collision::geometry::Line&, collision::geometry::TSphere<T>&, OutputVector*);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Line&, OutputVector*);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Point&, OutputVector*);
    template<class T> int computeIntersection(collision::geometry::Triangle&, collision::geometry::TSphere<T>&, OutputVector*);

protected:

    MinProximityIntersection* intersection;
};




template <class T>
bool MeshMinProximityIntersection::testIntersection(collision::geometry::Triangle& e2, collision::geometry::TSphere<T>& e1)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vec3 x13 = e2.p1()-e2.p2();
    const type::Vec3 x23 = e2.p1()-e2.p3();
    const type::Vec3 x03 = e2.p1()-e1.center();
    type::Matrix2 A;
    type::Vec2 b;
    A[0][0] = x13*x13;
    A[1][1] = x23*x23;
    A[0][1] = A[1][0] = x13*x23;
    b[0] = x13*x03;
    b[1] = x23*x03;
    const SReal det = type::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    //if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    type::Vec3 P,Q,PQ;
    P = e1.center();
    Q = e2.p1() - x13 * alpha - x23 * beta;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

template <class T>
int MeshMinProximityIntersection::computeIntersection(collision::geometry::Triangle& e2, collision::geometry::TSphere<T>& e1, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vec3 x13 = e2.p1()-e2.p2();
    const type::Vec3 x23 = e2.p1()-e2.p3();
    const type::Vec3 x03 = e2.p1()-e1.center();
    type::Matrix2 A;
    type::Vec2 b;
    A[0][0] = x13*x13;
    A[1][1] = x23*x23;
    A[0][1] = A[1][0] = x13*x23;
    b[0] = x13*x03;
    b[1] = x23*x03;
    const SReal det = type::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    //if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return 0;
    }

    const type::Vec3 P = e1.center();
    const type::Vec3 Q = e2.p1() - x13 * alpha - x23 * beta;
    const type::Vec3 QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    const SReal contactDist = intersection->getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->normal=QP;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        msg_warning(intersection) << "Null distance between contact detected";
        detection->normal= type::Vec3(1,0,0);
    }
    detection->value -= contactDist;
    detection->point[0]=Q;
    detection->point[1]=e1.getContactPointByNormal( detection->normal );
    return 1;
}

template <class T>
bool MeshMinProximityIntersection::testIntersection(collision::geometry::Line& e2, collision::geometry::TSphere<T>& e1)
{
    static_assert(std::is_same_v<collision::geometry::Line::Coord, typename collision::geometry::TSphere<T>::Coord>, "Data mismatch");
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const geometry::Line::Coord x32 = e2.p1()-e2.p2();
    const auto x31 = e1.center()-e2.p2();
    SReal A;
    SReal b;
    A = x32*x32;
    b = x32*x31;

    SReal alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return false;
    }

    type::Vec3 P,Q,PQ;
    P = e1.center();
    Q = e2.p1() - x32 * alpha;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

template <class T>
int MeshMinProximityIntersection::computeIntersection(collision::geometry::Line& e2, collision::geometry::TSphere<T>& e1, OutputVector* contacts)
{
    static_assert(std::is_same_v<collision::geometry::Line::Coord, typename collision::geometry::TSphere<T>::Coord>, "Data mismatch");
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const geometry::Line::Coord x32 = e2.p1()-e2.p2();
    const auto x31 = e1.center()-e2.p2();
    const geometry::Line::Coord::value_type A=x32*x32;
    const geometry::Line::Coord::value_type b=x32*x31;

    geometry::Line::Coord::value_type alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return 0;
    }

    const geometry::Line::Coord& P = e1.center();
    const geometry::Line::Coord Q = e2.p1() - x32 * alpha;
    const geometry::Line::Coord QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    const SReal contactDist = intersection->getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->normal=QP;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        msg_warning(intersection) << "Null distance between contact detected";
        detection->normal= type::Vec3(1,0,0);
    }
    detection->point[0]=Q;
    detection->point[1]=e1.getContactPointByNormal( detection->normal );
    detection->value -= contactDist;
    return 1;
}

template <class T>
bool MeshMinProximityIntersection::testIntersection(collision::geometry::TSphere<T>& e1, collision::geometry::Point& e2)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    type::Vec3 P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

template <class T>
int MeshMinProximityIntersection::computeIntersection(collision::geometry::TSphere<T>& e1, collision::geometry::Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    type::Vec3 P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;
    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const SReal contactDist = intersection->getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    //detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->id = e1.getIndex();
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        msg_warning(intersection) << "Null distance between contact detected";
        detection->normal= type::Vec3(1,0,0);
    }
    detection->value -= contactDist;
    detection->point[0]=e1.getContactPointByNormal( -detection->normal );
    detection->point[1]=Q;
    return 1;
}

} // namespace sofa::component::collision::detection::intersection
