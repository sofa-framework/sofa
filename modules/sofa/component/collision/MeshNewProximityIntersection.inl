/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_MESHNEWPROXIMITYINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_MESHNEWPROXIMITYINTERSECTION_INL

#include <sofa/helper/system/config.h>
#include <sofa/component/collision/MeshNewProximityIntersection.h>
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

inline int MeshNewProximityIntersection::doIntersectionLineLine(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2, OutputVector* contacts, int id)
{
    const Vector3 AB = p2-p1;
    const Vector3 CD = q2-q1;
    const Vector3 AC = q1-p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;

        if (alpha < 0.000001 || alpha > 0.999999 || beta < 0.000001 || beta > 0.999999 )
            return 0;
    }

    Vector3 p,q,pq;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;
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

inline int MeshNewProximityIntersection::doIntersectionLinePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q, OutputVector* contacts, int id, bool swapElems)
{
    const Vector3 AB = p2-p1;
    const Vector3 AQ = q -p1;
    double A;
    double b;
    A = AB*AB;
    b = AQ*AB;

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        //if (alpha < 0.000001 || alpha > 0.999999)
        //        return 0;
        if (alpha < 0.0) alpha = 0.0;
        else if (alpha > 1.0) alpha = 1.0;
    }

    Vector3 p,pq;
    p = p1 + AB * alpha;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    //detection->value -= contactDist;
    return 1;
}

inline int MeshNewProximityIntersection::doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& /*n*/, const Vector3& q, OutputVector* contacts, int id, bool swapElems)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = q -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        //if (alpha < 0.000001 ||
        //    beta  < 0.000001 ||
        //    alpha + beta  > 0.999999)
        //        return 0;
        if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
        {
            // nearest point is on an edge or corner
            // barycentric coordinate on AB
            double pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
            // barycentric coordinate on AC
            double pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
            if (pAB < 0.000001 && pAC < 0.0000001)
            {
                // closest point is A
                if (!(flags&TriangleModel::FLAG_P1)) return 0; // this corner is not considered
                alpha = 0.0;
                beta = 0.0;
            }
            else if (pAB < 0.999999 && beta < 0.000001)
            {
                // closest point is on AB
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = pAB;
                beta = 0.0;
            }
            else if (pAC < 0.999999 && alpha < 0.000001)
            {
                // closest point is on AC
                if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
                alpha = 0.0;
                beta = pAC;
            }
            else
            {
                // barycentric coordinate on BC
                // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
                double pBC = (b[1] - b[0] + A[0][0] - A[1][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
                if (pBC < 0.000001)
                {
                    // closest point is B
                    if (!(flags&TriangleModel::FLAG_P2)) return 0; // this edge is not considered
                    alpha = 1.0;
                    beta = 0.0;
                }
                else if (pBC > 0.999999)
                {
                    // closest point is C
                    if (!(flags&TriangleModel::FLAG_P3)) return 0; // this edge is not considered
                    alpha = 0.0;
                    beta = 1.0;
                }
                else
                {
                    // closest point is on BC
                    if (!(flags&TriangleModel::FLAG_E31)) return 0; // this edge is not considered
                    alpha = 1.0-pBC;
                    beta = pBC;
                }
            }
        }
    }

    Vector3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;

    //printf("\n normale : x = %f , y = %f, z = %f",detection->normal.x(),detection->normal.y(),detection->normal.z());
    //if (e2.getCollisionModel()->isStatic() && detection->normal * e2.n() < -0.95)
    //{ // The elements are interpenetrating
    //	detection->normal = -detection->normal;
    //	detection->value = -detection->value;
    //}
    //detection->value -= contactDist;
    return 1;
}

template<class Sphere>
bool MeshNewProximityIntersection::testIntersection(Sphere& e1, Point& e2)
{
    OutputVector contacts;
    const double alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();
    int n = intersection->doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), &contacts, -1);
    return n>0;
}

template<class Sphere>
int MeshNewProximityIntersection::computeIntersection(Sphere& e1, Point& e2, OutputVector* contacts)
{
    const double alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();
    int n = intersection->doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const double contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template<class Sphere>
bool MeshNewProximityIntersection::testIntersection(Line&, Sphere&)
{
    intersection->serr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Sphere)."<<intersection->sendl;
    return true;
}

template<class Sphere>
int MeshNewProximityIntersection::computeIntersection(Line& e1, Sphere& e2, OutputVector* contacts)
{
    const double alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.center(), contacts, e2.getIndex());
    if (n>0)
    {
        const double contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

template<class Sphere>
bool MeshNewProximityIntersection::testIntersection(Triangle&, Sphere&)
{
    intersection->serr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Sphere)."<<intersection->sendl;
    return true;
}

template<class Sphere>
int MeshNewProximityIntersection::computeIntersection(Triangle& e1, Sphere& e2, OutputVector* contacts)
{
    const double alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    const double dist2 = alarmDist*alarmDist;
    int n = doIntersectionTrianglePoint(dist2, e1.flags(),e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.center(), contacts, e2.getIndex());
    if (n>0)
    {
        const double contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
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
