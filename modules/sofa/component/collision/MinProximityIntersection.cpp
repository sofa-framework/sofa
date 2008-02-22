/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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

    if (det < -0.000000000001 || det > 0.000000000001)
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

    if (det < -0.000000000001 || det > 0.000000000001)
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

#ifdef DYNAMIC_CONE_ANGLE_COMPUTATION

    std::vector< std::pair <Vector3, Vector3> > neighborsTri;
    p.getTriangleNeighbors(neighborsTri);

    Vector3 nMean;
    nMean.clear();

    for (unsigned int i=0; i<neighborsTri.size(); i++)
    {
        Vector3 nCur = cross((neighborsTri[i].first) - pt, (neighborsTri[i].second) - pt);
        nCur.normalize();
        nMean += nCur;
    }

    nMean.normalize();

    std::vector<Vector3> neighborsPt;
    p.getLineNeighbors(neighborsPt);

    for (unsigned int i=0; i<neighborsPt.size(); i++)
    {
        Vector3 l = pt - neighborsPt[i];
        l.normalize();
        double computedAngleCone = (nMean * l) / 2;
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        //std::cout << "Point computedAngleCone = " << computedAngleCone << std::endl;

        if (l * PQ < -computedAngleCone*PQ.norm())
        {
            return false;
        }
    }

#else

    std::vector<Vector3> neighborsPt;
    p.getLineNeighbors(neighborsPt);

    for (unsigned int i=0; i<neighborsPt.size(); i++)
    {
        Vector3 l = pt - neighborsPt[i];
        if (l * PQ < -angleCone.getValue()*PQ.norm()*l.norm())
        {
            return false;
        }
    }

//	return true;

    std::vector< std::pair <Vector3, Vector3> > neighborsTri;
    p.getTriangleNeighbors(neighborsTri);

    Vector3 nMean;
    nMean.clear();

    for (unsigned int i=0; i<neighborsTri.size(); i++)
    {
        Vector3 nCur = cross((neighborsTri[i].first) - pt, (neighborsTri[i].second) - pt);
        nCur.normalize();
        nMean += nCur;
    }

    nMean.normalize();

#endif

    return ((nMean*PQ) >= 0.0);
}

bool MinProximityIntersection::testValidity(Line &l, const Vector3 &PQ)
{
    bool nMeanValue = false;
    Vector3 nMean;
    nMean.clear();

    const Vector3 &pt1 = l.p1();
    const Vector3 &pt2 = l.p2();

    Vector3 AB = pt2 - pt1;

#ifdef DYNAMIC_CONE_ANGLE_COMPUTATION

    Vector3 n1, n2;
    AB.normalize();

    // Right triangle
    const Vector3* tRight = l.tRight();
    if (tRight != NULL)
    {
        n1 = cross((*tRight)-pt1, pt2-pt1);
        nMean += n1;
        nMeanValue = true;
    }

    // Left triangle
    const Vector3* tLeft = l.tLeft();
    if (tLeft != NULL)
    {
        n2 = cross(pt2-pt1,(*tLeft)-pt1);
        nMean += n2;
        nMeanValue = true;
    }
    if (nMeanValue)
    {
        nMean.normalize();
        if ((nMean*PQ) < 0)
            return false;
    }

    if (!nMeanValue)
        return true;



    if (tRight != NULL)
    {
        n1.normalize();
        double computedAngleCone = (nMean * cross(n1, AB)) / 2;
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        //std::cout << "Right computedAngleCone = " << computedAngleCone << std::endl;

        if (cross(n1, AB)*PQ < -computedAngleCone*PQ.norm()*cross(n1, AB).norm())
            return false;
    }

    if (tLeft != NULL)
    {
        n2.normalize();
        double computedAngleCone = (nMean * cross(AB, n2)) / 2;
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        //std::cout << "Left computedAngleCone = " << computedAngleCone << std::endl;

        if (cross(AB, n2)*PQ < -computedAngleCone*PQ.norm()*cross(AB, n2).norm())
            return false;
    }

#else

    // Right triangle
    const Vector3* tRight = l.tRight();
    if (tRight != NULL)
    {
        Vector3 n1 = cross((*tRight)-pt1, pt2-pt1);
        nMean += n1;

        if (cross(n1, AB)*PQ < -angleCone.getValue()*PQ.norm()*cross(n1, AB).norm())
        {
            return false;
        }
    }

    // Left triangle
    const Vector3* tLeft = l.tLeft();
    if (tLeft != NULL)
    {
        Vector3 n2 = cross(pt2-pt1,(*tLeft)-pt1);
        nMean += n2;

        if (cross(AB, n2)*PQ < -angleCone.getValue()*PQ.norm()*cross(AB, n2).norm())
        {
            return false;
        }
    }

#endif

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

#ifdef _DEBUG
    glLineWidth(3);
    glBegin(GL_LINES);
    /*
    glColor4f(0,0,1,1);
    for (unsigned int i=0; i<IntersectionBuf.size(); i++)
    {
    	helper::gl::glVertexT(IntersectionBuf[i].first);
    	helper::gl::glVertexT(IntersectionBuf[i].second);
    }
    glColor4f(0,1,0,1);
    */
    for (unsigned int i=0; i<IntersectionBuf2.size(); i++)
    {
        helper::gl::glVertexT(IntersectionBuf2[i].first);
        helper::gl::glVertexT(IntersectionBuf2[i].second);
    }
    glEnd();
    glLineWidth(1);


    // fin: suppression du buffer
    IntersectionBuf.clear();
    IntersectionBuf2.clear();
#endif
}


} // namespace collision

} // namespace component

} // namespace sofa

