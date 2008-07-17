/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/MinProximityIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/componentmodel/collision/Intersection.inl>
//#include <sofa/component/collision/RayPickInteractor.h>
#include <iostream>
#include <algorithm>
#include <sofa/helper/gl/template.h>

#define DYNAMIC_CONE_ANGLE_COMPUTATION

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::collision;
using namespace helper;

SOFA_DECL_CLASS(MinProximityIntersection)

int MinProximityIntersectionClass = core::RegisterObject("TODO-MinProximityIntersection")
        .add< MinProximityIntersection >()
        ;

MinProximityIntersection::MinProximityIntersection()
    : useSphereTriangle(initData(&useSphereTriangle, true, "useSphereTriangle","activate Sphere-Triangle intersection tests"))
    , usePointPoint(initData(&usePointPoint, true, "usePointPoint","activate Point-Point intersection tests"))
    , alarmDistance(initData(&alarmDistance, 1.0, "alarmDistance","Proximity detection distance"))
    , contactDistance(initData(&contactDistance, 0.5, "contactDistance","Distance below which a contact is created"))
    , filterIntersection(initData(&filterIntersection, false, "filterIntersection","Intersections are filtered according to their orientation"))
    , angleCone(initData(&angleCone, 0.0, "angleCone","Filtering cone extension angle"))
    , coneFactor(initData(&coneFactor, 0.5, "coneFactor", "Factor for filtering cone angle computation"))
{
}

void MinProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, MinProximityIntersection>(this);
    if (usePointPoint.getValue())
        intersectors.add<PointModel, PointModel, MinProximityIntersection>(this);
    else
        intersectors.ignore<PointModel, PointModel>();
    intersectors.add<LineModel, LineModel, MinProximityIntersection>(this);
    intersectors.add<LineModel, PointModel, MinProximityIntersection>(this);
    intersectors.add<TriangleModel, PointModel, MinProximityIntersection>(this);
    intersectors.ignore<TriangleModel, LineModel>();
    intersectors.ignore<TriangleModel, TriangleModel>();
    if (useSphereTriangle.getValue())
    {
        intersectors.add<SphereModel, PointModel, MinProximityIntersection>(this);
        intersectors.add<TriangleModel, SphereModel, MinProximityIntersection>(this);
        intersectors.add<LineModel, SphereModel, MinProximityIntersection>(this);
    }
    else
    {
        intersectors.ignore<SphereModel, PointModel>();
        intersectors.ignore<LineModel, SphereModel>();
        intersectors.ignore<TriangleModel, SphereModel>();
    }
    intersectors.ignore<RayModel, PointModel>();
    intersectors.ignore<RayModel, LineModel>();
    intersectors.add<RayModel, TriangleModel, MinProximityIntersection>(this);
}

bool MinProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    const double alarmDist = getAlarmDistance() + cube1.getProximity() + cube2.getProximity();

    for (int i=0; i<3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i]> maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

int MinProximityIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}

bool MinProximityIntersection::testIntersection(Line& e1, Line& e2)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

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

    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

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
    {
        if (filterIntersection.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
            return true;
    }
    else
        return false;
}

int MinProximityIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

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
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

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
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    if (filterIntersection.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

#ifdef DETECTIONOUTPUT_FREEMOTION

    Vector3 Pfree,Qfree,ABfree,CDfree;
    ABfree = e1.p2Free()-e1.p1Free();
    CDfree = e2.p2Free()-e2.p1Free();
    Pfree = e1.p1Free() + ABfree * alpha;
    Qfree = e2.p1Free() + CDfree * beta;

#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[0]=Pfree;
    detection->freePoint[1]=Qfree;
#endif
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Triangle& e2, Point& e1)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

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
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

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
    {
        if (filterIntersection.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
            return true;
    }
    else
        return false;
}

int MinProximityIntersection::computeIntersection(Triangle& e2, Point& e1, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

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

    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return 0;
    }

    Vector3 P,Q,QP; //PQ
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    Vector3 PQ = Q-P;
    if (filterIntersection.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }


#ifdef DETECTIONOUTPUT_FREEMOTION

    Vector3 Pfree,Qfree,ABfree,ACfree;
    ABfree = e2.p2Free()-e2.p1Free();
    ACfree = e2.p3Free()-e2.p1Free();
    Pfree = e1.pFree();
    Qfree = e2.p1Free() + ABfree * alpha + ACfree * beta;

#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[0]=Qfree;
    detection->freePoint[1]=Pfree;
#endif
    detection->normal = QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Line& e2, Point& e1)
{

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;

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
    {
        if (filterIntersection.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
            return true;
    }
    else
        return false;
}

int MinProximityIntersection::computeIntersection(Line& e2, Point& e1, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const Vector3 AB = e2.p2()-e2.p1();
    const Vector3 AP = e1.p()-e2.p1();

    double A;
    double b;
    A = AB*AB;
    b = AP*AB;

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return 0;
    }

    Vector3 P,Q,QP;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    QP = P-Q;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    Vector3 PQ = Q-P;
    if (filterIntersection.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

#ifdef DETECTIONOUTPUT_FREEMOTION


    Vector3 ABfree = e2.p2Free()-e2.p1Free();
    Vector3 Pfree = e1.pFree();
    Vector3 Qfree = e2.p1Free() + ABfree * alpha;

#endif

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[0]=Qfree;
    detection->freePoint[1]=Pfree;
#endif
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Point& e1, Point& e2)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        if (filterIntersection.getValue())
        {
            if (!testValidity(e1, PQ))
                return false;

            Vector3 QP = -PQ;
            return testValidity(e2, QP);
        }
        else
            return true;
    }
    else
        return false;
}

int MinProximityIntersection::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;

#ifdef DETECTIONOUTPUT_FREEMOTION
    Vector3 Pfree,Qfree;
    Pfree = e1.pFree();
    Qfree = e2.pFree();
#endif


    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    if (filterIntersection.getValue())
    {
        if (!testValidity(e1, PQ))
            return 0;

        Vector3 QP = -PQ;

        if (!testValidity(e2, QP))
            return 0;
    }

    const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[0]=Pfree;
    detection->freePoint[1]=Qfree;
#endif
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}



bool MinProximityIntersection::testIntersection(Triangle& e2, Sphere& e1)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 x13 = e2.p1()-e2.p2();
    const Vector3 x23 = e2.p1()-e2.p3();
    const Vector3 x03 = e2.p1()-e1.center();
    Matrix2 A;
    Vector2 b;
    A[0][0] = x13*x13;
    A[1][1] = x23*x23;
    A[0][1] = A[1][0] = x13*x23;
    b[0] = x13*x03;
    b[1] = x23*x03;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    Vector3 P,Q,PQ;
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

int MinProximityIntersection::computeIntersection(Triangle& e2, Sphere& e1, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 x13 = e2.p1()-e2.p2();
    const Vector3 x23 = e2.p1()-e2.p3();
    const Vector3 x03 = e2.p1()-e1.center();
    Matrix2 A;
    Vector2 b;
    A[0][0] = x13*x13;
    A[1][1] = x23*x23;
    A[0][1] = A[1][0] = x13*x23;
    b[0] = x13*x03;
    b[1] = x23*x03;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    //if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return 0;
    }

    Vector3 P = e1.center();
    Vector3 Q = e2.p1() - x13 * alpha - x23 * beta;
    Vector3 QP = P-Q;
    Vector3 PQ = Q-P;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Line& e2, Sphere& e1)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 x32 = e2.p1()-e2.p2();
    const Vector3 x31 = e1.center()-e2.p2();
    double A;
    double b;
    A = x32*x32;
    b = x32*x31;

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return false;
    }

    Vector3 P,Q,PQ;
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

int MinProximityIntersection::computeIntersection(Line& e2, Sphere& e1, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    const Vector3 x32 = e2.p1()-e2.p2();
    const Vector3 x31 = e1.center()-e2.p2();
    double A;
    double b;
    A = x32*x32;
    b = x32*x31;

    double alpha = 0.5;

    //if (A < -0.000001 || A > 0.000001)
    {
        alpha = b/A;
        if (alpha < 0.000001 || alpha > 0.999999)
            return 0;
    }

    Vector3 P = e1.center();
    Vector3 Q = e2.p1() - x32 * alpha;
    Vector3 QP = P-Q;
    Vector3 PQ = Q-P;

    if (QP.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
    detection->id = e1.getIndex();
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=QP;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Sphere& e1, Point& e2)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
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

int MinProximityIntersection::computeIntersection(Sphere& e1, Point& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    Vector3 P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;
    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=PQ;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    detection->value -= contactDist;
    return 1;
}

bool MinProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    proximitySolver.NewComputation( &t2, A, B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        //std::cout<<"Collision between Line - Triangle"<<std::endl;
        return true;
    }
    else
        return false;
}

int MinProximityIntersection::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();


    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    proximitySolver.NewComputation( &t2, A,B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->id = t1.getIndex();
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[1] = P;
    detection->freePoint[0] = Q;
#endif
    detection->normal=-t2.n();
    detection->value = PQ.norm();
    detection->value -= contactDist;
    return 1;
}


bool MinProximityIntersection::testValidity(Point &p, const Vector3 &PQ)
{
    Vector3 pt = p.p();

    BaseMeshTopology* topology = p.getCollisionModel()->getTopology();
    helper::vector<Vector3>& x = *(p.getCollisionModel()->getMechanicalState()->getX());
    const helper::vector <unsigned int>& triangleVertexShell = topology->getTriangleVertexShell(p.getIndex());
    const helper::vector <unsigned int>& edgeVertexShell = topology->getEdgeVertexShell(p.getIndex());


    Vector3 nMean;

    for (unsigned int i=0; i<triangleVertexShell.size(); i++)
    {
        unsigned int t = triangleVertexShell[i];
        const fixed_array<unsigned int,3>& ptr = topology->getTriangle(t);
        Vector3 nCur = (x[ptr[1]]-x[ptr[0]]).cross(x[ptr[2]]-x[ptr[0]]);
        nCur.normalize();
        nMean += nCur;
    }

    if (triangleVertexShell.size()==0)
    {
        for (unsigned int i=0; i<edgeVertexShell.size(); i++)
        {
            unsigned int e = edgeVertexShell[i];
            const fixed_array<unsigned int,2>& ped = topology->getEdge(e);
            Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
            l.normalize();
            nMean += l;
        }
    }

    if (nMean.norm()> 0.0000000001)
        nMean.normalize();


    for (unsigned int i=0; i<edgeVertexShell.size(); i++)
    {
        unsigned int e = edgeVertexShell[i];
        const fixed_array<unsigned int,2>& ped = topology->getEdge(e);
        Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
        l.normalize();
        double computedAngleCone = dot(nMean , l) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();
        if (dot(l , PQ) < -computedAngleCone*PQ.norm())
            return false;
    }
    return true;

}

bool MinProximityIntersection::testValidity(Line &l, const Vector3 &PQ)
{
    Vector3 nMean;
    Vector3 n1, n2;
    Vector3 t1, t2;

    const Vector3 &pt1 = l.p1();
    const Vector3 &pt2 = l.p2();

    Vector3 AB = pt2 - pt1;
    AB.normalize();

    BaseMeshTopology* topology = l.getCollisionModel()->getTopology();
    helper::vector<Vector3>& x = *(l.getCollisionModel()->getMechanicalState()->getX());

    const sofa::helper::vector<unsigned int>& triangleEdgeShell = topology->getTriangleEdgeShell(l.getIndex());

    // filter if there are two triangles around the edge
    if (triangleEdgeShell.size() == 2)
    {
        // compute the normal of the triangle situated on the right
        const BaseMeshTopology::Triangle& triangleRight = topology->getTriangle(triangleEdgeShell[0]);
        n1 = cross(x[triangleRight[1]]-x[triangleRight[0]], x[triangleRight[2]]-x[triangleRight[0]]);
        n1.normalize();
        nMean = n1;
        t1 = -cross(n1, AB);
        t1.normalize(); // necessary ?

        // compute the normal of the triangle situated on the left
        const fixed_array<PointID,3>& triangleLeft = topology->getTriangle(triangleEdgeShell[1]);
        n2 = cross(x[triangleLeft[1]]-x[triangleLeft[0]], x[triangleLeft[2]]-x[triangleLeft[0]]);
        n2.normalize();
        nMean += n2;
        t2 = -cross(AB, n2);
        t2.normalize(); // necessary ?

        nMean.normalize();

        if ((nMean*PQ) < 0) // test
            return false;

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the right
        double computedAngleCone = (nMean * t1) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();

        if (t1*PQ < -computedAngleCone*PQ.norm())
            return false;

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the left
        computedAngleCone = (nMean * t2) * coneFactor.getValue();
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=angleCone.getValue();

        if (t2*PQ < -computedAngleCone*PQ.norm())
            return false;

    }
    return true;


}

bool MinProximityIntersection::testValidity(Triangle &t, const Vector3 &PQ)
{
    const Vector3& pt1 = t.p1();
    const Vector3& pt2 = t.p2();
    const Vector3& pt3 = t.p3();

    Vector3 n = cross(pt2-pt1,pt3-pt1);

    return ( (n*PQ) >= 0.0);
}


void MinProximityIntersection::draw()
{
    if (!getContext()->getShowCollisionModels())
        return;
}


} // namespace collision

} // namespace component

} // namespace sofa

