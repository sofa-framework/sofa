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
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/componentmodel/collision/Intersection.inl>
#include <sofa/component/collision/RayPickInteractor.h>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::collision;
using namespace helper;

SOFA_DECL_CLASS(NewProximityIntersection)

int NewProximityIntersectionClass = core::RegisterObject("TODO-NewProximityIntersection")
        .add< NewProximityIntersection >()
        ;

NewProximityIntersection::NewProximityIntersection()
    : alarmDistance(dataField(&alarmDistance, 1.0, "alarmDistance","Proximity detection distance"))
    , contactDistance(dataField(&contactDistance, 0.5, "contactDistance","Distance below which a contact is created"))
{
}

void NewProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, NewProximityIntersection, false>(this);
    intersectors.add<PointModel, PointModel, NewProximityIntersection, false>(this);
    intersectors.add<SphereModel, PointModel, NewProximityIntersection, true>(this);
    intersectors.add<SphereModel, SphereModel, NewProximityIntersection, false>(this);
    intersectors.add<LineModel, PointModel, NewProximityIntersection, true>(this);
    intersectors.add<LineModel, SphereModel, NewProximityIntersection, true>(this);
    intersectors.add<LineModel, LineModel, NewProximityIntersection, false>(this);
    intersectors.add<TriangleModel, PointModel, NewProximityIntersection, true>(this);
    intersectors.add<TriangleModel, SphereModel, NewProximityIntersection, true>(this);
    intersectors.add<TriangleModel, LineModel, NewProximityIntersection, true>(this);
    intersectors.add<TriangleModel, TriangleModel, NewProximityIntersection, false>(this);

    intersectors.add<RayModel, TriangleModel, NewProximityIntersection, true>(this);
    intersectors.add<RayPickInteractor, TriangleModel, NewProximityIntersection, true>(this);
}

int NewProximityIntersection::doIntersectionTrianglePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& /*n*/, const Vector3& q, DetectionOutputVector& contacts, bool swapElems)
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
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return 0;
    }

    Vector3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
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
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;

    //printf("\n normale : x = %f , y = %f, z = %f",detection->normal.x(),detection->normal.y(),detection->normal.z());
    //if (e2.getCollisionModel()->isStatic() && detection->normal * e2.n() < -0.95)
    //{ // The elements are interpenetrating
    //	detection->normal = -detection->normal;
    //	detection->distance = -detection->distance;
    //}
    //detection->distance -= contactDist;
    return 1;
}

int NewProximityIntersection::doIntersectionLineLine(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2, DetectionOutputVector& contacts)
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
        if (alpha < 0.000001 || alpha > 0.999999 ||
            beta  < 0.000001 || beta  > 0.999999 )
            return 0;
    }

    Vector3 p,q,pq;
    p = p1 + AB * alpha;
    q = q1 + CD * beta;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    //detection->distance -= contactDist;
    return 1;
}

int NewProximityIntersection::doIntersectionLinePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q, DetectionOutputVector& contacts, bool swapElems)
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
        if (alpha < 0.000001 || alpha > 0.999999)
            return 0;
    }

    Vector3 p,pq;
    p = p1 + AB * alpha;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);

    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
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
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    //detection->distance -= contactDist;
    return 1;
}

int NewProximityIntersection::doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, DetectionOutputVector& contacts)
{
    Vector3 pq;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    //const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);
    //detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    //detection->distance -= contactDist;
    return 1;
}

//static NewProximityIntersection* proximityInstance = NULL;

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* NewProximityIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2)
{
    //proximityInstance = this;
    return this->DiscreteIntersection::findIntersector(object1, object2);
}



bool NewProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
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

int NewProximityIntersection::computeIntersection(Cube&, Cube&, DetectionOutputVector& /*contacts*/)
{
    return 0; /// \todo
}

bool NewProximityIntersection::testIntersection(Point& e1, Point& e2)
{
    DetectionOutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), contacts);
    return n>0;
}

int NewProximityIntersection::computeIntersection(Point& e1, Point& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), contacts);
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Sphere& e1, Point& e2)
{
    DetectionOutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), contacts);
    return n>0;
}

int NewProximityIntersection::computeIntersection(Sphere& e1, Point& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.p(), contacts);
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Sphere& e1, Sphere& e2)
{
    DetectionOutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts);
    return n>0;
}

int NewProximityIntersection::computeIntersection(Sphere& e1, Sphere& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts);
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Line&, Point&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Point).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Line& e1, Point& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.p(), contacts);
    if (n==0)
    {
        n += doIntersectionPointPoint(alarmDist*alarmDist, e1.p1(),e2.p(), contacts);
        n += doIntersectionPointPoint(alarmDist*alarmDist, e1.p2(),e2.p(), contacts);
    }
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Line&, Sphere&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Sphere).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Line& e1, Sphere& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(),e1.p2(), e2.center(), contacts);
    if (n==0)
    {
        n += doIntersectionPointPoint(alarmDist*alarmDist, e1.p1(),e2.center(), contacts);
        n += doIntersectionPointPoint(alarmDist*alarmDist, e1.p2(),e2.center(), contacts);
    }
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Line&, Line&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Line).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Line& e1, Line& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    int n = doIntersectionLineLine(dist2, e1.p1(),e1.p2(), e2.p1(),e2.p2(), contacts);
    if (n==0)
    {
        n += doIntersectionLinePoint(dist2, e1.p1(),e1.p2(), e2.p1(), contacts, false);
        n += doIntersectionLinePoint(dist2, e1.p1(),e1.p2(), e2.p2(), contacts, false);
        n += doIntersectionLinePoint(dist2, e2.p1(),e2.p2(), e1.p1(), contacts, true);
        n += doIntersectionLinePoint(dist2, e2.p1(),e2.p2(), e1.p2(), contacts, true);
        if (n==0)
        {
            n += doIntersectionPointPoint(dist2, e1.p1(),e2.p1(), contacts);
            n += doIntersectionPointPoint(dist2, e1.p1(),e2.p2(), contacts);
            n += doIntersectionPointPoint(dist2, e1.p2(),e2.p1(), contacts);
            n += doIntersectionPointPoint(dist2, e1.p2(),e2.p2(), contacts);
        }
    }
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Triangle&, Point&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Point).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Triangle& e1, Point& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    int n = doIntersectionTrianglePoint(dist2, e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.p(), contacts);
    if (n==0)
    {
        const int f1 = e1.flags();
        if (f1&TriangleModel::FLAG_E12)
            n += doIntersectionLinePoint(dist2, e1.p1(),e1.p2(), e2.p(), contacts);
        if (f1&TriangleModel::FLAG_E23)
            n += doIntersectionLinePoint(dist2, e1.p2(),e1.p3(), e2.p(), contacts);
        if (f1&TriangleModel::FLAG_E31)
            n += doIntersectionLinePoint(dist2, e1.p3(),e1.p1(), e2.p(), contacts);

        if (f1&TriangleModel::FLAG_P1)
            n += doIntersectionPointPoint(dist2, e1.p1(), e2.p(), contacts);
        if (f1&TriangleModel::FLAG_P2)
            n += doIntersectionPointPoint(dist2, e1.p2(), e2.p(), contacts);
        if (f1&TriangleModel::FLAG_P3)
            n += doIntersectionPointPoint(dist2, e1.p3(), e2.p(), contacts);
    }
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Triangle&, Sphere&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Sphere).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Triangle& e1, Sphere& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e2.r();
    const double dist2 = alarmDist*alarmDist;
    int n = doIntersectionTrianglePoint(dist2, e1.p1(),e1.p2(),e1.p3(),e1.n(), e2.center(), contacts);
    if (n==0)
    {
        const int f1 = e1.flags();
        if (f1&TriangleModel::FLAG_E12)
            n += doIntersectionLinePoint(dist2, e1.p1(),e1.p2(), e2.center(), contacts);
        if (f1&TriangleModel::FLAG_E23)
            n += doIntersectionLinePoint(dist2, e1.p2(),e1.p3(), e2.center(), contacts);
        if (f1&TriangleModel::FLAG_E31)
            n += doIntersectionLinePoint(dist2, e1.p3(),e1.p1(), e2.center(), contacts);

        if (f1&TriangleModel::FLAG_P1)
            n += doIntersectionPointPoint(dist2, e1.p1(), e2.center(), contacts);
        if (f1&TriangleModel::FLAG_P2)
            n += doIntersectionPointPoint(dist2, e1.p2(), e2.center(), contacts);
        if (f1&TriangleModel::FLAG_P3)
            n += doIntersectionPointPoint(dist2, e1.p3(), e2.center(), contacts);
    }
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e2.r();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}

bool NewProximityIntersection::testIntersection(Triangle&, Line&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle& e1, Line& e2).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Triangle& e1, Line& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();

    const int f1 = e1.flags();

    int n = 0;

    if (f1&TriangleModel::FLAG_P1)
    {
        n += doIntersectionPointPoint(dist2, p1, q1, contacts);
        n += doIntersectionPointPoint(dist2, p1, q2, contacts);
        n += doIntersectionLinePoint(dist2, q1,q2, p1, contacts, true);
    }
    if (f1&TriangleModel::FLAG_P2)
    {
        n += doIntersectionPointPoint(dist2, p2, q1, contacts);
        n += doIntersectionPointPoint(dist2, p2, q2, contacts);
        n += doIntersectionLinePoint(dist2, q1,q2, p2, contacts, true);
    }
    if (f1&TriangleModel::FLAG_P3)
    {
        n += doIntersectionPointPoint(dist2, p3, q1, contacts);
        n += doIntersectionPointPoint(dist2, p3, q2, contacts);
        n += doIntersectionLinePoint(dist2, q1,q2, p3, contacts, true);
    }

    if (f1&TriangleModel::FLAG_E12)
    {
        n += doIntersectionLineLine (dist2, p1,p2, q1,q2, contacts);
        n += doIntersectionLinePoint(dist2, p1,p2, q1, contacts, false);
        n += doIntersectionLinePoint(dist2, p1,p2, q2, contacts, false);
    }
    if (f1&TriangleModel::FLAG_E23)
    {
        n += doIntersectionLineLine (dist2, p2,p3, q1,q2, contacts);
        n += doIntersectionLinePoint(dist2, p2,p3, q1, contacts, false);
        n += doIntersectionLinePoint(dist2, p2,p3, q2, contacts, false);
    }
    if (f1&TriangleModel::FLAG_E31)
    {
        n += doIntersectionLineLine (dist2, p3,p1, q1,q2, contacts);
        n += doIntersectionLinePoint(dist2, p3,p1, q1, contacts, false);
        n += doIntersectionLinePoint(dist2, p3,p1, q2, contacts, false);
    }

    n += doIntersectionTrianglePoint(dist2, p1,p2,p3,pn, q1, contacts, false);
    n += doIntersectionTrianglePoint(dist2, p1,p2,p3,pn, q2, contacts, false);

    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}


bool NewProximityIntersection::testIntersection(Triangle&, Triangle&)
{
    std::cerr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle& e1, Triangle& e2).\n";
    return true;
}

int NewProximityIntersection::computeIntersection(Triangle& e1, Triangle& e2, DetectionOutputVector& contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
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

    int n = 0;

    // Point-Point tests
    if (f1&TriangleModel::FLAG_P1)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionPointPoint(dist2, p1, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionPointPoint(dist2, p1, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionPointPoint(dist2, p1, q3, contacts);
    }
    if (f1&TriangleModel::FLAG_P2)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionPointPoint(dist2, p2, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionPointPoint(dist2, p2, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionPointPoint(dist2, p2, q3, contacts);
    }
    if (f1&TriangleModel::FLAG_P3)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionPointPoint(dist2, p3, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionPointPoint(dist2, p3, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionPointPoint(dist2, p3, q3, contacts);
    }

    // Triangle-Point tests
    if (f1&TriangleModel::FLAG_P1)
        n += doIntersectionTrianglePoint(dist2, q1,q2,q3,qn, p1, contacts, true);
    if (f1&TriangleModel::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, q1,q2,q3,qn, p2, contacts, true);
    if (f1&TriangleModel::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, q1,q2,q3,qn, p3, contacts, true);

    if (f2&TriangleModel::FLAG_P1)
        n += doIntersectionTrianglePoint(dist2, p1,p2,p3,pn, q1, contacts, false);
    if (f2&TriangleModel::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, p1,p2,p3,pn, q2, contacts, false);
    if (f2&TriangleModel::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, p1,p2,p3,pn, q3, contacts, false);

    // Line-Line tests
    if (f1&TriangleModel::FLAG_E12)
    {
        if (f2&TriangleModel::FLAG_E12) n += doIntersectionLineLine(dist2, p1,p2, q1,q2, contacts);
        if (f2&TriangleModel::FLAG_E23) n += doIntersectionLineLine(dist2, p1,p2, q2,q3, contacts);
        if (f2&TriangleModel::FLAG_E31) n += doIntersectionLineLine(dist2, p1,p2, q3,q1, contacts);
    }
    if (f1&TriangleModel::FLAG_E23)
    {
        if (f2&TriangleModel::FLAG_E12) n += doIntersectionLineLine(dist2, p2,p3, q1,q2, contacts);
        if (f2&TriangleModel::FLAG_E23) n += doIntersectionLineLine(dist2, p2,p3, q2,q3, contacts);
        if (f2&TriangleModel::FLAG_E31) n += doIntersectionLineLine(dist2, p2,p3, q3,q1, contacts);
    }
    if (f1&TriangleModel::FLAG_E31)
    {
        if (f2&TriangleModel::FLAG_E12) n += doIntersectionLineLine(dist2, p3,p1, q1,q2, contacts);
        if (f2&TriangleModel::FLAG_E23) n += doIntersectionLineLine(dist2, p3,p1, q2,q3, contacts);
        if (f2&TriangleModel::FLAG_E31) n += doIntersectionLineLine(dist2, p3,p1, q3,q1, contacts);
    }

    // Line-Point tests
    if (f1&TriangleModel::FLAG_E12)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, p1,p2, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, p1,p2, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, p1,p2, q3, contacts);
    }
    if (f1&TriangleModel::FLAG_E23)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, p2,p3, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, p2,p3, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, p2,p3, q3, contacts);
    }
    if (f1&TriangleModel::FLAG_E31)
    {
        if (f2&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, p3,p1, q1, contacts);
        if (f2&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, p3,p1, q2, contacts);
        if (f2&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, p3,p1, q3, contacts);
    }
    if (f2&TriangleModel::FLAG_E12)
    {
        if (f1&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, q1,q2, p1, contacts, true);
        if (f1&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, q1,q2, p2, contacts, true);
        if (f1&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, q1,q2, p3, contacts, true);
    }
    if (f2&TriangleModel::FLAG_E23)
    {
        if (f1&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, q2,q3, p1, contacts, true);
        if (f1&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, q2,q3, p2, contacts, true);
        if (f1&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, q2,q3, p3, contacts, true);
    }
    if (f2&TriangleModel::FLAG_E31)
    {
        if (f1&TriangleModel::FLAG_P1) n += doIntersectionLinePoint(dist2, q3,q1, p1, contacts, true);
        if (f1&TriangleModel::FLAG_P2) n += doIntersectionLinePoint(dist2, q3,q1, p2, contacts, true);
        if (f1&TriangleModel::FLAG_P3) n += doIntersectionLinePoint(dist2, q3,q1, p3, contacts, true);
    }

    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (DetectionOutputVector::iterator detection = contacts.end()-n; detection != contacts.end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->distance -= contactDist;
        }
    }
    return n;
}








bool NewProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;
    const double alarmDist = 0.01; //getAlarmDistance();

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

int NewProximityIntersection::computeIntersection(Ray &t1, Triangle &t2, DetectionOutputVector& contacts)
{
    const double alarmDist = 0.01; //getAlarmDistance();

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

    const double contactDist = 0.01; //getAlarmDistance();
    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t2, t1);
    detection->point[0]=P;
    detection->point[1]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[0] = P;
    detection->freePoint[0] = Q;
#endif
    detection->normal=t2.n();
    detection->distance = PQ.norm();
    detection->distance -= contactDist;
    return 1;
}

} // namespace collision

} // namespace component

} // namespace sofa

