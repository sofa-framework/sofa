/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaGeneralMeshCollision/MeshMinProximityIntersection.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <SofaBaseCollision/DiscreteIntersection.h>
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

SOFA_DECL_CLASS(MeshMinProximityIntersection)

IntersectorCreator<MinProximityIntersection, MeshMinProximityIntersection> MeshMinProximityIntersectors("Mesh");

MeshMinProximityIntersection::MeshMinProximityIntersection(MinProximityIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        if (intersection->usePointPoint.getValue())
            intersection->intersectors.add<PointModel, PointModel, MeshMinProximityIntersection>(this);
        else
            intersection->intersectors.ignore<PointModel, PointModel>();

        if(intersection->useLinePoint.getValue())
            intersection->intersectors.add<LineModel, PointModel, MeshMinProximityIntersection>(this);
        else
            intersection->intersectors.ignore<LineModel, PointModel>();

        if(intersection->useLineLine.getValue())
            intersection->intersectors.add<LineModel, LineModel, MeshMinProximityIntersection>(this);
        else
            intersection->intersectors.ignore<LineModel, LineModel>();

        intersection->intersectors.add<TriangleModel, PointModel, MeshMinProximityIntersection>(this);
        intersection->intersectors.ignore<TriangleModel, LineModel>();
        intersection->intersectors.ignore<TriangleModel, TriangleModel>();
        intersection->intersectors.add<CapsuleModel, TriangleModel, MeshMinProximityIntersection>(this);
        intersection->intersectors.add<CapsuleModel, LineModel, MeshMinProximityIntersection>(this);

        if (intersection->useSphereTriangle.getValue())
        {
            intersection->intersectors.add<SphereModel, PointModel, MeshMinProximityIntersection>(this);
            intersection->intersectors.add<RigidSphereModel, PointModel, MeshMinProximityIntersection>(this);
            intersection->intersectors.add<TriangleModel, SphereModel, MeshMinProximityIntersection>(this);
            intersection->intersectors.add<TriangleModel, RigidSphereModel, MeshMinProximityIntersection>(this);
            intersection->intersectors.add<LineModel, SphereModel, MeshMinProximityIntersection>(this);
            intersection->intersectors.add<LineModel, RigidSphereModel, MeshMinProximityIntersection>(this);
        }
        else
        {
            intersection->intersectors.ignore<SphereModel, PointModel>();
            intersection->intersectors.ignore<RigidSphereModel, PointModel>();
            intersection->intersectors.ignore<LineModel, SphereModel>();
            intersection->intersectors.ignore<LineModel, RigidSphereModel>();
            intersection->intersectors.ignore<TriangleModel, SphereModel>();
            intersection->intersectors.ignore<TriangleModel, RigidSphereModel>();
        }

//    intersection->intersectors.ignore<RayModel, PointModel>();
//    intersection->intersectors.ignore<RayModel, LineModel>();
    }
}

bool MeshMinProximityIntersection::testIntersection(Line& e1, Line& e2)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e1.p2()-e1.p1();
    const Vector3 CD = e2.p2()-e2.p1();
    const Vector3 AC = e2.p1()-e1.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;

    const SReal det = determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    if (det < -1.0e-18 || det > 1.0e-18)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 || alpha > 0.999999 ||
            beta  < 0.000001 || beta  > 0.999999 )
            return false;
    }

    Vector3 PQ = AC + CD * beta - AB * alpha;

    if (PQ.norm2() < alarmDist*alarmDist)
        return true;
    else
        return false;
}

int MeshMinProximityIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e1.p2()-e1.p1();
    const Vector3 CD = e2.p2()-e2.p1();
    const Vector3 AC = e2.p1()-e1.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = CD*CD;
    A[0][1] = A[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const SReal det = determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    if (det < -1.0e-15 || det > 1.0e-15)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 || alpha > 0.999999 ||
            beta  < 0.000001 || beta  > 0.999999 )
            return 0;
    }

    Vector3 P,Q,PQ;
    P = e1.p1() + AB * alpha;
    Q = e2.p1() + CD * beta;

    PQ  = Q - P;
    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree,ABfree,CDfree;
        ABfree = e1.p2Free()-e1.p1Free();
        CDfree = e2.p2Free()-e2.p1Free();
        Pfree = e1.p1Free() + ABfree * alpha;
        Qfree = e2.p1Free() + CDfree * beta;

        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }
#endif

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0] = P;
    detection->point[1] = Q;
    detection->normal = PQ;
    detection->value = detection->normal.norm();

    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->value -= contactDist;

    return 1;
}

bool MeshMinProximityIntersection::testIntersection(Triangle& e2, Point& e1)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

    // We want to find alpha,beta so that:
    // AQ = AB*alpha+AC*beta
    // PQ.AB = 0 and PQ.AC = 0
    // (AQ-AP).AB = 0 and (AQ-AP).AC = 0
    // AQ.AB = AP.AB and AQ.AC = AP.AC
    //
    // (AB*alpha+AC*beta).AB = AP.AB and
    // (AB*alpha+AC*beta).AC = AP.AC
    //
    // AB.AB*alpha + AC.AB*beta = AP.AB and
    // AB.AC*alpha + AC.AC*beta = AP.AC
    //
    // A . [alpha beta] = b
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;
    const SReal det = determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    const Vector3 PQ = AB * alpha + AC * beta - AP;

    if (PQ.norm2() < alarmDist*alarmDist)
        return true;
    else
        return false;
}

int MeshMinProximityIntersection::computeIntersection(Triangle& e2, Point& e1, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AC = e2.p3()-e2.p1();
    const Vector3 AP = e1.p() -e2.p1();
    Matrix2 A;
    Vector2 b;

    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AP*AB;
    b[1] = AP*AC;

    const SReal det = determinant(A);

    SReal alpha = 0.5;
    SReal beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return 0;
//        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
//        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
//        if (alpha < 0 ||
//            beta  < 0 ||
//            alpha + beta  > 1)
//            return 0;
    }

    Vector3 P,Q,QP; //PQ
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    //Vector3 PQ = Q-P;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree,Qfree,ABfree,ACfree;
        ABfree = e2.p2Free()-e2.p1Free();
        ACfree = e2.p3Free()-e2.p1Free();
        Pfree = e1.pFree();
        Qfree = e2.p1Free() + ABfree * alpha + ACfree * beta;

        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal = QP;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->value -= contactDist;

    if(intersection->getUseSurfaceNormals())
    {
        int normalIndex = e2.getIndex();
        detection->normal = e2.model->getNormals()[normalIndex];
    }    

    return 1;
}

bool MeshMinProximityIntersection::testIntersection(Line& e2, Point& e1)
{

    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    SReal A;
    SReal b;
    A = AB*AB;
    b = AP*AB;

    SReal alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return false;
    }

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
        return true;
    else
        return false;
}

int MeshMinProximityIntersection::computeIntersection(Line& e2, Point& e1, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    SReal A;
    SReal b;
    A = AB*AB;
    b = AP*AB;

    SReal alpha = 0.5;

    Vector3 P,Q,QP;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;

        if (alpha <= 0.0){
            Q = e2.p1();
        }
        else if (alpha >= 1.0){
            Q = e2.p2();
//            alpha = 1.0;
        }
        else{
            Q = e2.p1() + AB * alpha;
        }
    }

    P = e1.p();
    QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 ABfree = e2.p2Free()-e2.p1Free();
        Vector3 Pfree = e1.pFree();
        Vector3 Qfree = e2.p1Free() + ABfree * alpha;
        detection->freePoint[0] = Qfree;
        detection->freePoint[1] = Pfree;
    }
#endif

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->value -= contactDist;

    return 1;
}

bool MeshMinProximityIntersection::testIntersection(Point& e1, Point& e2)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
        return true;
    else
        return false;
}

int MeshMinProximityIntersection::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

#ifdef DETECTIONOUTPUT_FREEMOTION
    if (e1.hasFreePosition() && e2.hasFreePosition())
    {
        Vector3 Pfree, Qfree;
        Pfree = e1.pFree();
        Qfree = e2.pFree();

        detection->freePoint[0] = Pfree;
        detection->freePoint[1] = Qfree;
    }
#endif

    const SReal contactDist = intersection->getContactDistance() + e1.getProximity() + e2.getProximity();

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();

    if(detection->value>1e-15)
    {
        detection->normal /= detection->value;
    }
    else
    {
        intersection->serr<<"WARNING: null distance between contact detected"<<intersection->sendl;
        detection->normal= Vector3(1,0,0);
    }
    detection->value -= contactDist;

    return 1;
}



int MeshMinProximityIntersection::computeIntersection(Capsule & cap,Triangle & tri,OutputVector* contacts){
    return MeshIntTool::computeIntersection(cap,tri,intersection->getAlarmDistance(),intersection->getContactDistance(),contacts);
}

int MeshMinProximityIntersection::computeIntersection(Capsule & cap,Line & lin,OutputVector* contacts){
    return MeshIntTool::computeIntersection(cap,lin,intersection->getAlarmDistance(),intersection->getContactDistance(),contacts);
}

bool MeshMinProximityIntersection::testIntersection(Capsule&,Triangle&){
    return true;
}

bool MeshMinProximityIntersection::testIntersection(Capsule&,Line&){
    return true;
}

} // namespace collision

} // namespace component

} // namespace sofa

