#include "Common/config.h"
#include "Common/FnDispatcher.inl"
#include "DiscreteIntersection.h"
#include "Common/ObjectFactory.h"
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
using namespace DiscreteIntersections;

namespace Common
{
template<>
void create(DiscreteIntersection*& obj, ObjectDescription* /*arg*/)
{
    obj = new DiscreteIntersection();
}
}

SOFA_DECL_CLASS(DiscreteIntersection)

Creator<ObjectFactory, DiscreteIntersection> DiscreteIntersectionClass("DiscreteIntersection");

DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeModel,     CubeModel,     intersectionCubeCube,         distCorrectionCubeCube,         false>();
    intersectors.add<SphereModel,   SphereModel,   intersectionSphereSphere,     distCorrectionSphereSphere,     false>();
    intersectors.add<SphereModel,   RayModel,      intersectionSphereRay,        distCorrectionSphereRay,        true>();
    intersectors.add<SphereModel,   RayPickInteractor,      intersectionSphereRay,        distCorrectionSphereRay,        true>();
    intersectors.add<SphereTreeModel, SphereTreeModel, intersectionSingleSphereSingleSphere, distCorrectionSingleSphereSingleSphere,     false>();
    //intersectors.add<SphereTreeModel, SphereModel, intersectionSingleSphereSingleSphere, distCorrectionSingleSphereSingleSphere,     false>();
    //intersectors.add<SphereModel,   TriangleModel, intersectionSphereTriangle,   distCorrectionSphereTriangle, true>();
    //intersectors.add<TriangleModel, TriangleModel, intersectionTriangleTriangle, distCorrectionTriangleTriangle, false>();
}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(Abstract::CollisionModel* object1, Abstract::CollisionModel* object2)
{
    return intersectors.get(object1, object2);
}


namespace DiscreteIntersections
{

bool intersectionSphereSphere(Sphere& sph1, Sphere& sph2)
{
    //std::cout<<"Collision between Sphere - Sphere"<<std::endl;
    Vector3 sph1Pos(sph1.center());
    Vector3 sph2Pos(sph2.center());
    double radius1 = sph1.r(), radius2 = sph2.r();
    Vector3 tmp = sph1Pos - sph2Pos;

    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}

bool intersectionSingleSphereSingleSphere(SingleSphere& sph1, SingleSphere& sph2)
{
    //std::cout<<"Collision between Sphere - Sphere"<<std::endl;
    //Vector3 sph1Pos = sph1.getCenter();
    //Vector3 sph2Pos = sph2.getCenter();

    Vector3 sph1Pos(sph1.center());
    Vector3 sph2Pos(sph2.center());

    double radius1 = sph1.r(), radius2 = sph2.r();
    Vector3 tmp = sph1Pos - sph2Pos;

    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}


bool intersectionCubeCube(Cube& cube1, Cube& cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    for (int i=0; i<3; i++)
    {
        if (minVect1[i] > maxVect2[i] || minVect2[i] > maxVect1[i])
            return false;
    }

    //std::cout << "Box <"<<minVect1[0]<<","<<minVect1[1]<<","<<minVect1[2]<<">-<"<<maxVect1[0]<<","<<maxVect1[1]<<","<<maxVect1[2]
    //  <<"> collide with Box "<<minVect2[0]<<","<<minVect2[1]<<","<<minVect2[2]<<">-<"<<maxVect2[0]<<","<<maxVect2[1]<<","<<maxVect2[2]<<">"<<std::endl;
    return true;
}

bool intersectionSphereRay(Sphere& sph1, Ray& ray2)
{
    //std::cout<<"Collision between Sphere - Ray"<<std::endl;

    const Vector3 sph1Pos(sph1.center());
    const double radius1 = sph1.r();
    const Vector3 ray2Origin(ray2.origin());
    const Vector3 ray2Direction(ray2.direction());
    const double length2 = ray2.l();
    const Vector3 tmp = sph1Pos - ray2Origin;
    const double rayPos = tmp*ray2Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

//bool intersectionSphereTriangle(Sphere&, Triangle&)
//{
//	std::cout<<"Collision between Sphere - Triangle"<<std::endl;
//	return false;
//}

//bool intersectionTriangleTriangle(Triangle& t1, Triangle& t2)
//{
//	std::cout<<"Collision between Triangle - Triangle"<<std::endl;
//	return false;
//}

DetectionOutput* distCorrectionCubeCube(Cube&, Cube&)
{
    return NULL; /// \todo
}

DetectionOutput* distCorrectionSphereSphere(Sphere& sph1, Sphere& sph2)
{
    DetectionOutput *detection = new DetectionOutput();
    detection->normal = sph2.center() - sph1.center();
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->distance = distSph1Sph2 - (sph1.r() + sph2.r());
    detection->elem.first = sph1;
    detection->elem.second = sph2;

    return detection;
}

DetectionOutput* distCorrectionSingleSphereSingleSphere(SingleSphere& sph1, SingleSphere& sph2)
{
    DetectionOutput *detection = new DetectionOutput();
    detection->normal = sph2.center() - sph1.center();
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->distance = distSph1Sph2 - (sph1.r() + sph2.r());
    detection->elem.first = sph1;
    detection->elem.second = sph2;

    return detection;
}
DetectionOutput* distCorrectionSphereRay(Sphere& sph1, Ray& ray2)
{
    const Vector3 sph1Pos(sph1.center());
    const double radius1 = sph1.r();
    const Vector3 ray2Origin(ray2.origin());
    const Vector3 ray2Direction(ray2.direction());
    const double length2 = ray2.l();
    const Vector3 tmp = sph1Pos - ray2Origin;
    const double rayPos = tmp*ray2Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist = sqrt(tmp.norm2() - (rayPosInside*rayPosInside));

    DetectionOutput *detection = new DetectionOutput();

    detection->point[1] = ray2Origin + ray2Direction*rayPosInside;
    detection->normal = detection->point[1] - sph1Pos;
    detection->normal /= dist;
    detection->point[0] = sph1Pos + detection->normal * radius1;
    detection->distance = dist - radius1;
    detection->elem.first = sph1;
    detection->elem.second = ray2;

    return detection;
}

//DetectionOutput* distCorrectionSphereTriangle(Sphere&, Triangle&)
//{
//	std::cout<<"Distance correction between Sphere - Triangle"<<std::endl;
//	return new DetectionOutput();
//}

//DetectionOutput* distCorrectionTriangleTriangle(Triangle&, Triangle&)
//{
//	std::cout<<"Distance correction between Triangle - Triangle"<<std::endl;
//	return new DetectionOutput();
//}

} // namespace DiscreteIntersections

} // namespace Components

} // namespace Sofa
