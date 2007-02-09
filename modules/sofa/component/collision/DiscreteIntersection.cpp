#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/collision/Intersection.inl>
#include <sofa/component/collision/RayPickInteractor.h>
#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/component/collision/proximity.h>
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
using namespace DiscreteIntersections;

SOFA_DECL_CLASS(DiscreteIntersection)

int DiscreteIntersectionClass = core::RegisterObject("TODO")
        .add< DiscreteIntersection >()
        ;


DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeModel,     CubeModel,     intersectionCubeCube,         distCorrectionCubeCube,         false>();
    intersectors.add<SphereModel,   SphereModel,   intersectionSphereSphere,     distCorrectionSphereSphere,     false>();
    intersectors.add<SphereModel,   RayModel,      intersectionSphereRay,        distCorrectionSphereRay,        true>();
    intersectors.add<SphereModel,   RayPickInteractor,      intersectionSphereRay,        distCorrectionSphereRay,        true>();
    intersectors.add<SphereTreeModel, RayPickInteractor, intersectionSingleSphereRay, distCorrectionSingleSphereRay, true>();
    intersectors.add<SphereTreeModel, SphereTreeModel, intersectionSingleSphereSingleSphere, distCorrectionSingleSphereSingleSphere,     false>();
    intersectors.add<SphereTreeModel, CubeModel, intersectionSingleSphereCube, distCorrectionSingleSphereCube, true> ();
    intersectors.add<SphereTreeModel, TriangleModel, intersectionSingleSphereTriangle, distCorrectionSingleSphereTriangle, true> ();
    //intersectors.add<SphereTreeModel, SphereModel, intersectionSingleSphereSingleSphere, distCorrectionSingleSphereSingleSphere,     false>();
    //intersectors.add<SphereModel,   TriangleModel, intersectionSphereTriangle,   distCorrectionSphereTriangle, true>();
    //intersectors.add<TriangleModel, TriangleModel, intersectionTriangleTriangle, distCorrectionTriangleTriangle, false>();
}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2)
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
    Vector3 sph1Pos(sph1.center());
    Vector3 sph2Pos(sph2.center());

    double radius1 = sph1.r(), radius2 = sph2.r();
    Vector3 tmp = sph1Pos - sph2Pos;

    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}

bool intersectionSingleSphereCube( SingleSphere& sph1, Cube& cube)
{
    // Values of the "aligned" bounding box
    Vector3 Bmin = cube.minVect();
    Vector3 Bmax = cube.maxVect();
    // Center of sphere
    Vector3 ctr(sph1.center());
    // Square of radius
    double r2 = sph1.r()*sph1.r();
    // Distance
    double dmin = 0;

    for ( int i = 0; i<3; i++)
    {
        if ( ctr[i] < Bmin[i] ) 	dmin += (ctr[i]-Bmin[i])*(ctr[i]-Bmin[i]);
        else if ( ctr[i] > Bmax[i] ) dmin += (ctr[i]-Bmax[i])*(ctr[i]-Bmax[i]);
    }

    if (dmin <= r2 ) return true;
    else return false;
}

bool intersectionSingleSphereTriangle( SingleSphere& sph, Triangle& triangle)
{
    // todo
    double EPSILON = 0.00001;
    //Vertices of the triangle:
    Vector3 p0 = triangle.p1();
    Vector3 p1 = triangle.p2();
    Vector3 p2 = triangle.p3();

    // Center of the sphere
    const Vector3 sphCenter(sph.center());
    // Radius of the sphere
    const double r = sph.r();

    //Normal to the plane (plane spanned by tree points of the triangle)
    Vector3 normal = cross( (p1 - p0), (p2 - p0) );
    normal.normalize();

    //Distance from the center of the sphere to the plane.
    double distance = sphCenter*normal - normal*p0;

    //Projection of the center of the sphere onto the plane
    Vector3 projPoint = sphCenter - normal*distance;

    //Distance correction in case is negative.
    if (distance < 0.0)
        distance = -distance;

    //Distance to the sphere:
    distance -= r;

    //If the distance is positive, the point has been proyected outside
    //the sphere and hence the plane does not intersect the sphere
    //and so the triangle (that spanned the plane) cannot be inside the sphere.
    if (distance  > EPSILON)
    {
        return false;
    }

    //However, if the plane has intersected the sphere, then it is
    //neccesary to check if the proyected point "projPoint" is inside
    //the triangle.
#define SAMESIDE(ap1,ap2,ap3,ap4) (((cross((ap4-ap3),(ap1-ap3))) * (cross((ap4-ap3),(ap2-ap3)))) >= 0)
    if ( (SAMESIDE(projPoint,p0,p1,p2) && SAMESIDE(projPoint,p1,p0,p2) && SAMESIDE(projPoint,p2,p0,p1)))
    {
        return true;
    }
#undef SAMESIDE
    return false;
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


bool intersectionSingleSphereRay(SingleSphere& sph1, Ray& ray2)
{
    // Center of the sphere
    const Vector3 sph1Pos(sph1.center());
    // Radius of the sphere
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

DetectionOutput* distCorrectionSingleSphereTriangle( SingleSphere& sph, Triangle& triangle)
{
    double EPSILON = 0.00001;
    //Vertices of the triangle:
    Vector3 p0 = triangle.p1();
    Vector3 p1 = triangle.p2();
    Vector3 p2 = triangle.p3();

    // Center of the sphere
    const Vector3 sphCenter(sph.center());
    // Radius of the sphere
    const double r = sph.r();

    //Normal to the plane (plane spanned by tree points of the triangle)
    Vector3 normal = cross( (p1 - p0), (p2 - p0) );
    normal.normalize();

    //Distance from the center of the sphere to the plane.
    double distance = sphCenter*normal - normal*p0;

    //Projection of the center of the sphere onto the plane
    Vector3 projPoint = sphCenter - normal*distance;

    //Distance correction in case is negative.
    if (distance < 0.0)
        distance = -distance;

    //Distance to the sphere:
    distance -= r;

    //If the distance is positive, the point has been proyected outside
    //the sphere and hence the plane does not intersect the sphere
    //and so the triangle (that spanned the plane) cannot be inside the sphere.
    if (distance  > EPSILON)
    {
        return NULL;
    }

    //However, if the plane has intersected the sphere, then it is
    //neccesary to check if the proyected point "projPoint" is inside
    //the triangle.
#define SAMESIDE(ap1,ap2,ap3,ap4) (((cross((ap4-ap3),(ap1-ap3))) * (cross((ap4-ap3),(ap2-ap3)))) >= 0)
    if ( (SAMESIDE(projPoint,p0,p1,p2) && SAMESIDE(projPoint,p1,p0,p2) && SAMESIDE(projPoint,p2,p0,p1)))
    {
        DetectionOutput *detection = new DetectionOutput();
        detection->normal = -normal;
        detection->point[1] = projPoint;
        detection->point[0] = sph.center() - detection->normal * sph.r();

        detection->distance = -distance;
        //detection->elem.first = triangle;
        //detection->elem.second = sph;
        detection->elem.first = sph;
        detection->elem.second = triangle;
        return detection;
    }
#undef SAMESIDE

    //// The projected sphere center is not in the triangle. Verify if
    //// the edges are colliding the sphere (check if they are secant to the sphere)
    // RayModel edges;
    ////Edge 0
    // Vector3 dir = p1 - p0;
    // double length = dir.norm();
    // edges.addRay(p0,dir,length);
    ////Edge1
    // dir = p1 - p2;
    // length = dir.norm();
    // edges.addRay(p1,dir,length);
    // //Edge2
    // dir = p2 - p0;
    // length = dir.norm();
    // edges.addRay(p2,dir,length);
    //
    // detection = distCorrectionSingleSphereRay( sph,edges.getRay(0));
    //if ( detection != NULL )
    //{
    //	detection->elem.first = triangle;
    //	detection->elem.second = sph;
    //	return detection;
    //}

    //detection = distCorrectionSingleSphereRay( sph,edges.getRay(1));
    //if ( detection != NULL )
    //{
    //	detection->elem.first = triangle;
    //	detection->elem.second = sph;
    //	return detection;
    //}
    // detection = distCorrectionSingleSphereRay( sph,edges.getRay(2));
    //	if ( detection != NULL )
    //{
    //	detection->elem.first = triangle;
    //	detection->elem.second = sph;
    //	return detection;
    //}

    return NULL; // No intersection: passed all tests for intersections !
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

DetectionOutput* distCorrectionSingleSphereCube(SingleSphere& /*sph1*/, Cube& /*cube*/)
{
    //to do
    return NULL;
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

DetectionOutput* distCorrectionSingleSphereRay(SingleSphere& sph1, Ray& ray2)
{
    const Vector3 sph1Pos(sph1.center());
    const double radius1 = sph1.r();
    const Vector3 ray2Origin(ray2.origin());
    const Vector3 ray2Direction(ray2.direction());
    const double length2 = ray2.l();
    const Vector3 tmp = sph1Pos - ray2Origin;
    const double rayPos = tmp*ray2Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    double dist =  sqrt(tmp.norm2() - (rayPosInside*rayPosInside));

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

} // namespace collision

} // namespace component

} // namespace sofa

