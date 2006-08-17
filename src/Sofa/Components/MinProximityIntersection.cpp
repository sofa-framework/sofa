#include "Common/config.h"
#include "MinProximityIntersection.h"
#include "Common/ObjectFactory.h"
#include "Proximity/proximity.h"
#include "Common/Mat.h"
#include "Common/Vec.h"
#include "Collision/Intersection.inl"
#include "RayPickInteractor.h"

#include <iostream>
#include <algorithm>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;
using namespace Proximity;
using namespace MinProximityIntersections;

namespace Common
{
template<>
void create(MinProximityIntersection*& obj, ObjectDescription* arg)
{
    obj = new MinProximityIntersection(
        (atoi(arg->getAttribute("useSphereTriangle","1"))!=0)
    );
    obj->setAlarmDistance(atof(arg->getAttribute("alarmDistance","1.0")));
    obj->setContactDistance(atof(arg->getAttribute("contactDistance","0.5")));
}
}

SOFA_DECL_CLASS(MinProximityIntersection)

Creator<ObjectFactory, MinProximityIntersection> MinProximityIntersectionClass("MinProximityIntersection");

MinProximityIntersection::MinProximityIntersection(bool useSphereTriangle
                                                  )
    : alarmDistance(1.0), contactDistance(0.5)
{
    intersectors.add<CubeModel, CubeModel, intersectionCubeCube, distCorrectionCubeCube, false>();
    intersectors.ignore<TriangleModel, TriangleModel, false>();
    intersectors.ignore<LineModel, TriangleModel, true>();
    intersectors.add<LineModel, LineModel, intersectionLineLine, distCorrectionLineLine, false>();
    intersectors.add<PointModel, TriangleModel, intersectionPointTriangle, distCorrectionPointTriangle, true>();
    intersectors.add<PointModel, LineModel, intersectionPointLine, distCorrectionPointLine, true>();
    intersectors.add<PointModel, PointModel, intersectionPointPoint, distCorrectionPointPoint, false>();

    if (useSphereTriangle)
    {
        intersectors.add<SphereModel, TriangleModel, intersectionSphereTriangle, distCorrectionSphereTriangle, true>();
        intersectors.add<SphereModel, LineModel, intersectionSphereLine, distCorrectionSphereLine, true>();
        intersectors.add<SphereModel, PointModel, intersectionSpherePoint, distCorrectionSpherePoint, true>();
    }
    else
    {
        intersectors.ignore<SphereModel, TriangleModel, true>();
        intersectors.ignore<SphereModel, LineModel, true>();
        intersectors.ignore<SphereModel, PointModel, true>();
    }
    intersectors.add<RayModel, TriangleModel, intersectionRayTriangle, distCorrectionRayTriangle, true>();
    intersectors.add<RayPickInteractor, TriangleModel, intersectionRayTriangle, distCorrectionRayTriangle, true>();
}

/// \todo Use a better way to transmit parameters

static MinProximityIntersection* proximityInstance = NULL;

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* MinProximityIntersection::findIntersector(Abstract::CollisionModel* object1, Abstract::CollisionModel* object2)
{
    proximityInstance = this;
    return this->DiscreteIntersection::findIntersector(object1, object2);
}

namespace MinProximityIntersections
{

bool intersectionCubeCube(Cube &cube1, Cube &cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();
    const double alarmDist = proximityInstance->getAlarmDistance();

    for (int i=0; i<3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i]> maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

DetectionOutput* distCorrectionCubeCube(Cube&, Cube&)
{
    return NULL; /// \todo
}

bool intersectionLineLine(Line& e1, Line& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance();
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

    if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 || alpha > 0.999999 ||
            beta  < 0.000001 || beta  > 0.999999 )
            return false;
    }

    //Vector3 PQ = e2.p1() + CD * beta - (e1.p1() + AB * alpha);
    Vector3 PQ = AC + CD * beta - AB * alpha;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

DetectionOutput* distCorrectionLineLine(Line& e1, Line& e2)
{
    const double contactDist = proximityInstance->getContactDistance();
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

    if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 || alpha > 0.999999 ||
            beta  < 0.000001 || beta  > 0.999999 )
            return NULL;
    }

    Vector3 P,Q,PQ;
    P = e1.p1() + AB * alpha;
    Q = e2.p1() + CD * beta;
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e1, e2);
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=Q-P;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}

bool intersectionPointTriangle(Point& e1, Triangle& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance();
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

    if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

DetectionOutput* distCorrectionPointTriangle(Point& e1, Triangle& e2)
{
    const double contactDist = proximityInstance->getContactDistance();
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

    if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return false;
    }

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha + AC * beta;
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e2, e1);
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=P-Q;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    if (e2.getCollisionModel()->isStatic() && detection->normal * e2.n() < -0.95)
    {
        // The elements are interpenetrating
        detection->normal = -detection->normal;
        detection->distance = -detection->distance;
    }
    detection->distance -= contactDist;
    return detection;
}

bool intersectionPointLine(Point& e1, Line& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance();
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
        return true;
    }
    else
        return false;
}

DetectionOutput* distCorrectionPointLine(Point& e1, Line& e2)
{
    const double contactDist = proximityInstance->getContactDistance();
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
            return NULL;
    }

    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p1() + AB * alpha;
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e2, e1);
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=P-Q;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}

bool intersectionPointPoint(Point& e1, Point& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance();
    Vector3 PQ = e2.p()-e1.p();

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        return true;
    }
    else
        return false;
}

DetectionOutput* distCorrectionPointPoint(Point& e1, Point& e2)
{
    const double contactDist = proximityInstance->getContactDistance();
    Vector3 P,Q,PQ;
    P = e1.p();
    Q = e2.p();
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e1, e2);
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=Q-P;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}



bool intersectionSphereTriangle(Sphere& e1, Triangle& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance() + e1.r();
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

    if (det < -0.000001 || det > 0.000001)
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

DetectionOutput* distCorrectionSphereTriangle(Sphere& e1, Triangle& e2)
{
    const double contactDist = proximityInstance->getContactDistance() + e1.r();
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

    if (det < -0.000001 || det > 0.000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        if (alpha < 0.000001 ||
            beta  < 0.000001 ||
            alpha + beta  > 0.999999)
            return NULL;
    }

    Vector3 P,Q;
    P = e1.center();
    Q = e2.p1() - x13 * alpha - x23 * beta;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e2, e1);
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=P-Q;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}

bool intersectionSphereLine(Sphere& e1, Line& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance() + e1.r();
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

DetectionOutput* distCorrectionSphereLine(Sphere& e1, Line& e2)
{
    const double contactDist = proximityInstance->getContactDistance() + e1.r();
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
            return NULL;
    }

    Vector3 P,Q,PQ;
    P = e1.center();
    Q = e2.p1() - x32 * alpha;
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e2, e1);
    detection->point[0]=Q;
    detection->point[1]=P;
    detection->normal=P-Q;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}

bool intersectionSpherePoint(Sphere& e1, Point& e2)
{
    const double alarmDist = proximityInstance->getAlarmDistance() + e1.r();
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

DetectionOutput* distCorrectionSpherePoint(Sphere& e1, Point& e2)
{
    const double contactDist = proximityInstance->getContactDistance() + e1.r();
    Vector3 P,Q,PQ;
    P = e1.center();
    Q = e2.p();
    PQ = Q-P;

    DetectionOutput *detection = new DetectionOutput();
    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(e1, e2);
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=Q-P;
    detection->distance = detection->normal.norm();
    detection->normal /= detection->distance;
    detection->distance -= contactDist;
    return detection;
}




bool intersectionRayTriangle(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;
    const double alarmDist = 0.01; //proximityInstance->getAlarmDistance();

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

Collision::DetectionOutput* distCorrectionRayTriangle(Ray &t1, Triangle &t2)
{
    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    Vector3 P,Q;
    static DistanceSegTri proximitySolver;
    DetectionOutput *detection = new DetectionOutput();
    const double contactDist = proximityInstance->getContactDistance();

    proximitySolver.NewComputation( &t2, A,B,P,Q);

    detection->elem = std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator>(t2, t1);
    detection->point[0]=P;
    detection->point[1]=Q;
    detection->normal=t2.n();
    detection->distance = (Q-P).norm();
    detection->distance -= contactDist;
    return detection;
}

} // namespace MinProximityIntersections

} // namespace Components

} // namespace Sofa
