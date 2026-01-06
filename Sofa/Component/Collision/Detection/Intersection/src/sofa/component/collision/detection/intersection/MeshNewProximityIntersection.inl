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
#include <sofa/component/collision/detection/intersection/MeshNewProximityIntersection.h>
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.inl>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/core/collision/Intersection.inl>

namespace sofa::component::collision::detection::intersection
{

inline int MeshNewProximityIntersection::doIntersectionLineLine(SReal dist2, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& q1, const type::Vec3& q2, OutputVector* contacts, int id, const type::Vec3& /*n*/, bool /*useNormal*/)
{  
    const auto AB = p2 - p1;
    const auto CD = q2 - q1;
    const auto AC = q1 - p1;
    type::Matrix2 A;
    type::Vec2 b;
    A(0,0) = AB * AB;
    A(1,1) = CD * CD;
    A(0,1) = A(1,0) = -CD * AB;
    b[0] = AB * AC;
    b[1] = -CD * AC;
    const double det = type::determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0] * A(1,1) - b[1] * A(0,1)) / det;
        beta = (b[1] * A(0,0) - b[0] * A(1,0)) / det;

        if (alpha < 0.000001 || alpha > 0.999999 || beta < 0.000001 || beta > 0.999999)
            return 0;
    }

    type::Vec3 p,q;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;

    const auto pq = q-p;
    const auto norm2 = pq.norm2();

    if (norm2 >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->value = helper::rsqrt(norm2);
    detection->normal = pq / detection->value;
    //detection->value -= contactDist;
    return 1;
}

inline int MeshNewProximityIntersection::doIntersectionLinePoint(SReal dist2, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems)
{
    const auto AB = p2 - p1;
    const auto AQ = q - p1;
    double A;
    double b;
    A = AB * AB;
    b = AQ * AB;

    double alpha = 0.5;

    alpha = b / A;
    if (alpha < 0.0) alpha = 0.0;
    else if (alpha > 1.0) alpha = 1.0;

    const auto p = p1 + AB * alpha;
    const auto pq = q-p;
    const auto norm2 = pq.norm2();
    if (norm2 >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);

    detection->id = id;
    detection->value = helper::rsqrt(norm2);
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq / detection->value;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq / detection->value;
    }

    return 1;
}

inline int MeshNewProximityIntersection::doIntersectionTrianglePoint2(SReal dist2, int flags, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& /*n*/, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems)
{
    using collision::geometry::TriangleCollisionModel;

    const type::Vec3 AB = p2-p1;
    const type::Vec3 AC = p3-p1;
    const type::Vec3 AQ = q -p1;
    type::Matrix2 A;
    type::Vec2 b;
    A(0,0) = AB*AB;
    A(1,1) = AC*AC;
    A(0,1) = A(1,0) = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = type::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    alpha = (b[0]*A(1,1) - b[1]*A(0,1))/det;
    beta  = (b[1]*A(0,0) - b[0]*A(1,0))/det;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        const SReal pAB = b[0] / A(0,0); // AQ*AB / AB*AB
        // barycentric coordinate on AC
        const SReal pAC = b[1] / A(1,1); // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            // closest point is A
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            // closest point is on AB
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            // closest point is on AC
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            const SReal pBC = (b[1] - b[0] + A(0,0) - A(0,1)) / (A(0,0) + A(1,1) - 2*A(0,1)); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                // closest point is B
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)) return 0; // this edge is not considered
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }

    type::Vec3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    const SReal norm2 = pq.norm2();
    if (pq.norm2() >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->id = id;
    detection->value = helper::rsqrt(norm2);
    if (swapElems)
    {
        detection->point[1]=p;
        detection->normal = -pq / detection->value;
    }
    else
    {
        detection->point[0]=p;
        detection->normal = pq / detection->value;
    }
    return 1;
}



inline int MeshNewProximityIntersection::doIntersectionTrianglePoint(SReal dist2, int flags, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& /*n*/, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems, bool /*useNormal*/)
{
    using collision::geometry::TriangleCollisionModel;

    const type::Vec3 AB = p2-p1;
    const type::Vec3 AC = p3-p1;
    const type::Vec3 AQ = q -p1;
    type::Matrix2 A;
    type::Vec2 b;
    A(0,0) = AB*AB;
    A(1,1) = AC*AC;
    A(0,1) = A(1,0) = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = type::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;
    const SReal epsilon=std::numeric_limits<SReal>::epsilon();

    alpha = (b[0]*A(1,1) - b[1]*A(0,1))/det;
    beta  = (b[1]*A(0,0) - b[0]*A(1,0))/det;
    if (alpha < epsilon || beta < epsilon || alpha + beta > 1 - epsilon)
    {
            //return 0;
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
            const SReal pAB = b[0] / A(0,0); // AQ*AB / AB*AB
        // barycentric coordinate on AC
            const SReal pAC = b[1] / A(1,1); // AQ*AB / AB*AB
        if (pAB < epsilon && pAC < epsilon)
        {
            // closest point is A
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 1 - epsilon && pAB >= epsilon && beta < epsilon)
        {
            // closest point is on AB
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 1 - epsilon && pAC >= epsilon && alpha < epsilon)
        {
            // closest point is on AC
            if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            const SReal pBC = (b[1] - b[0] + A(0,0) - A(0,1)) / (A(0,0) + A(1,1) - 2*A(0,1)); // BQ*BC / BC*BC
            if (pBC < epsilon)
            {
                // closest point is B
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)) return 0; // this edge is not considered
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 1 - epsilon)
            {
                // closest point is C
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                if (!(flags&TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E23)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }

    type::Vec3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    const SReal norm2 = pq.norm2();
    if (pq.norm2() >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->id = id;
    detection->value = helper::rsqrt(norm2);

    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq / detection->value;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq / detection->value;
    }
    return 1;
}

template <class T>
bool MeshNewProximityIntersection::testIntersection(collision::geometry::TSphere<T>& e1, collision::geometry::Point& e2, const core::collision::Intersection* currentIntersection)
{
    OutputVector contacts;
    const double alarmDist = currentIntersection->getAlarmDistance() + e1.getContactDistance() + e2.getContactDistance() + e1.r();

    // By design, MeshNewProximityIntersection is supposed to work only with NewProximityIntersection
    const auto* currentNewProxIntersection = static_cast<const NewProximityIntersection*>(currentIntersection);

    const int n = currentNewProxIntersection->doIntersectionPointPoint(alarmDist * alarmDist, e1.center(), e2.p(), &contacts, -1);
    return n > 0;
}

template<class T>
int MeshNewProximityIntersection::computeIntersection(collision::geometry::TSphere<T>& e1, collision::geometry::Point& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    const SReal alarmDist = currentIntersection->getAlarmDistance() + e1.getContactDistance() + e2.getContactDistance() + e1.r();

    // By design, MeshNewProximityIntersection is supposed to work only with NewProximityIntersection
    const auto* currentNewProxIntersection = static_cast<const NewProximityIntersection*>(currentIntersection);

    const int n = currentNewProxIntersection->doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = currentIntersection->getContactDistance() + e1.getContactDistance() + e2.getContactDistance() + e1.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template <class T>
bool MeshNewProximityIntersection::testIntersection(collision::geometry::Line& e1, collision::geometry::TSphere<T>& e2, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(e1);
    SOFA_UNUSED(e2);

    msg_warning(currentIntersection) << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Sphere).";
    return true;
}

template<class T>
int MeshNewProximityIntersection::computeIntersection(collision::geometry::Line& e1, collision::geometry::TSphere<T>& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    const SReal alarmDist = currentIntersection->getAlarmDistance() + e1.getContactDistance() + e2.getContactDistance() + e2.r();
    const int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.center(), contacts, e2.getIndex());
    if (n>0)
    {
        const SReal contactDist = currentIntersection->getContactDistance() + e1.getContactDistance() + e2.getContactDistance() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template <class T>
bool MeshNewProximityIntersection::testIntersection(collision::geometry::Triangle& e1, collision::geometry::TSphere<T>& e2, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(e1);
    SOFA_UNUSED(e2);
    SOFA_UNUSED(currentIntersection);

    msg_warning(currentIntersection) << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Sphere).";
    return true;
}

template<class T>
int MeshNewProximityIntersection::computeIntersection(collision::geometry::Triangle& e1, collision::geometry::TSphere<T>& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    const int flags = e1.flags();
    const SReal alarmDist = currentIntersection->getAlarmDistance() + e1.getContactDistance() + e2.getContactDistance() + e2.r();
    const SReal dist2 = alarmDist*alarmDist;

    const type::Vec3 AB = e1.p2() - e1.p1();
    const type::Vec3 AC = e1.p3() - e1.p1();
    const type::Vec3 AQ = e2.center() - e1.p1();
    type::Matrix2 A;
    type::Vec2 b;
    A(0,0) = AB*AB;
    A(1,1) = AC*AC;
    A(0,1) = A(1,0) = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const SReal det = type::determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    alpha = (b[0]*A(1,1) - b[1]*A(0,1))/det;
    beta  = (b[1]*A(0,0) - b[0]*A(1,0))/det;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        const SReal pAB = b[0] / A(0,0); // AQ*AB / AB*AB
        // barycentric coordinate on AC
        const SReal pAC = b[1] / A(1,1); // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            // closest point is A
            if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            // closest point is on AB
            if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            // closest point is on AC
            if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E12)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            const SReal pBC = (b[1] - b[0] + A(0,0) - A(0,1)) / (A(0,0) + A(1,1) - 2*A(0,1)); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                // closest point is B
                if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P2)) return 0; // this edge is not considered
                alpha = 1.0;
                beta = 0.0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_P3)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = 1.0;
            }
            else
            {
                // closest point is on BC
                if (!(flags&collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::FLAG_E31)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
            }
        }
    }

    type::Vec3 p, pq;
    p = e1.p1() + AB * alpha + AC * beta;
    pq = e2.center() - p;
    const SReal norm2 = pq.norm2();
    if (pq.norm2() >= dist2)
        return 0;

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = e2.getIndex();
    detection->value = helper::rsqrt(norm2) ;

    if(detection->value>1e-15)
    {
        detection->normal = pq / detection->value;
    }
    else
    {
        msg_warning(currentIntersection) <<"Null distance between contact detected";
        detection->normal= type::Vec3(1,0,0);
    }

    detection->value -= (currentIntersection->getContactDistance() + e1.getContactDistance() + e2.getContactDistance() + e2.r());
    detection->point[0]=p;
    detection->point[1]= e2.getContactPointByNormal(detection->normal);

    return 1;
}

} // namespace sofa::component::collision::detection::intersection
