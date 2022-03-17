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
#include <sofa/component/collision/model/SphereModel.h>
#include <sofa/component/collision/model/TriangleModel.h>
#include <sofa/component/collision/model/LineModel.h>
#include <sofa/component/collision/model/PointModel.h>
#include <sofa/component/collision/model/CubeModel.h>

namespace sofa::component::collision::detection::intersection
{

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshMinProximityIntersection : public core::collision::BaseIntersector
{
    typedef MinProximityIntersection::OutputVector OutputVector;

public:
    MeshMinProximityIntersection(MinProximityIntersection* object, bool addSelf=true);

    bool testIntersection(model::Point&, model::Point&);
    template<class T> bool testIntersection(model::TSphere<T>&, model::Point&);
    bool testIntersection(model::Line&, model::Point&);
    template<class T> bool testIntersection(model::Line&, model::TSphere<T>&);
    bool testIntersection(model::Line&, model::Line&);
    bool testIntersection(model::Triangle&, model::Point&);
    template<class T> bool testIntersection(model::Triangle&, model::TSphere<T>&);

    int computeIntersection(model::Point&, model::Point&, OutputVector*);        
    template<class T> int computeIntersection(model::TSphere<T>&, model::Point&, OutputVector*);
    int computeIntersection(model::Line&, model::Point&, OutputVector*);
    template<class T> int computeIntersection(model::Line&, model::TSphere<T>&, OutputVector*);
    int computeIntersection(model::Line&, model::Line&, OutputVector*);
    int computeIntersection(model::Triangle&, model::Point&, OutputVector*);
    template<class T> int computeIntersection(model::Triangle&, model::TSphere<T>&, OutputVector*);

protected:

    MinProximityIntersection* intersection;
};




template <class T>
bool MeshMinProximityIntersection::testIntersection(model::Triangle& e2, model::TSphere<T>& e1)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vector3 x13 = e2.p1()-e2.p2();
    const type::Vector3 x23 = e2.p1()-e2.p3();
    const type::Vector3 x03 = e2.p1()-e1.center();
    type::Matrix2 A;
    type::Vector2 b;
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

    type::Vector3 P,Q,PQ;
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
int MeshMinProximityIntersection::computeIntersection(model::Triangle& e2, model::TSphere<T>& e1, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vector3 x13 = e2.p1()-e2.p2();
    const type::Vector3 x23 = e2.p1()-e2.p3();
    const type::Vector3 x03 = e2.p1()-e1.center();
    type::Matrix2 A;
    type::Vector2 b;
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

    type::Vector3 P = e1.center();
    type::Vector3 Q = e2.p1() - x13 * alpha - x23 * beta;
    type::Vector3 QP = P-Q;
    //Vector3 PQ = Q-P;

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
        detection->normal= type::Vector3(1,0,0);
    }
    detection->value -= contactDist;
    detection->point[0]=Q;
    detection->point[1]=e1.getContactPointByNormal( detection->normal );
    return 1;
}

template <class T>
bool MeshMinProximityIntersection::testIntersection(model::Line& e2, model::TSphere<T>& e1)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vector3 x32 = e2.p1()-e2.p2();
    const type::Vector3 x31 = e1.center()-e2.p2();
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

    type::Vector3 P,Q,PQ;
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
int MeshMinProximityIntersection::computeIntersection(model::Line& e2, model::TSphere<T>& e1, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const type::Vector3 x32 = e2.p1()-e2.p2();
    const type::Vector3 x31 = e1.center()-e2.p2();
    SReal A;
    SReal b;
    A = x32*x32;
    b = x32*x31;

    SReal alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return 0;
    }

    type::Vector3 P = e1.center();
    type::Vector3 Q = e2.p1() - x32 * alpha;
    type::Vector3 QP = P-Q;
    //Vector3 PQ = Q-P;

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
        detection->normal= type::Vector3(1,0,0);
    }
    detection->point[0]=Q;
    detection->point[1]=e1.getContactPointByNormal( detection->normal );
    detection->value -= contactDist;
    return 1;
}

template <class T>
bool MeshMinProximityIntersection::testIntersection(model::TSphere<T>& e1, model::Point& e2)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    type::Vector3 P,Q,PQ;
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
int MeshMinProximityIntersection::computeIntersection(model::TSphere<T>& e1, model::Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    type::Vector3 P,Q,PQ;
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
        detection->normal= type::Vector3(1,0,0);
    }
    detection->value -= contactDist;
    detection->point[0]=e1.getContactPointByNormal( -detection->normal );
    detection->point[1]=Q;
    return 1;
}

} // namespace sofa::component::collision::detection::intersection
