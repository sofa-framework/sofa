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
#ifndef SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/collision/Intersection.inl>
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

template<class Sphere>
bool DiscreteIntersection::testIntersection(Sphere& sph1, Sphere& sph2)
{
    //std::cout<<"Collision between Sphere - Sphere"<<std::endl;
    typename Sphere::Coord sph1Pos(sph1.center());
    typename Sphere::Coord sph2Pos(sph2.center());
    typename Sphere::Real radius1 = sph1.r(), radius2 = sph2.r();
    typename Sphere::Coord tmp = sph1Pos - sph2Pos;
    return (tmp.norm2() < (radius1 + radius2) * (radius1 + radius2));
}

template<class Sphere>
bool DiscreteIntersection::testIntersection( Sphere& sph1, Cube& cube)
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
        if ( ctr[i] < Bmin[i] )      dmin += (ctr[i]-Bmin[i])*(ctr[i]-Bmin[i]);
        else if ( ctr[i] > Bmax[i] ) dmin += (ctr[i]-Bmax[i])*(ctr[i]-Bmax[i]);
    }

    return (dmin <= r2 );
}

template<class Sphere>
bool DiscreteIntersection::testIntersection( Sphere& sph, Triangle& triangle)
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
        return false;
    }

    //However, if the plane has intersected the sphere, then it is
    //necessary to check if the projected point "projPoint" is inside
    //the triangle.
#define SAMESIDE(ap1,ap2,ap3,ap4) (((cross((ap4-ap3),(ap1-ap3))) * (cross((ap4-ap3),(ap2-ap3)))) >= 0)
    if ( (SAMESIDE(projPoint,p0,p1,p2) && SAMESIDE(projPoint,p1,p0,p2) && SAMESIDE(projPoint,p2,p0,p1)))
    {
        return true;
    }
#undef SAMESIDE
    return false;
}

template<class Sphere>
bool DiscreteIntersection::testIntersection(Sphere& sph1, Ray& ray2)
{
    //std::cout<<"intersectionSphereRay: Collision between Sphere - Ray"<<std::endl;
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

template<class Sphere>
int DiscreteIntersection::computeIntersection( Sphere& sph, Triangle& triangle, DetectionOutputVector& contacts)
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
        return 0;
    }

    //However, if the plane has intersected the sphere, then it is
    //neccesary to check if the proyected point "projPoint" is inside
    //the triangle.
#define SAMESIDE(ap1,ap2,ap3,ap4) (((cross((ap4-ap3),(ap1-ap3))) * (cross((ap4-ap3),(ap2-ap3)))) >= 0)
    if ( (SAMESIDE(projPoint,p0,p1,p2) && SAMESIDE(projPoint,p1,p0,p2) && SAMESIDE(projPoint,p2,p0,p1)))
    {
        contacts.resize(contacts.size()+1);
        DetectionOutput *detection = &*(contacts.end()-1);
        detection->normal = -normal;
        detection->point[1] = projPoint;
        detection->point[0] = sph.center() - detection->normal * sph.r();

        detection->distance = -distance;
        //detection->elem.first = triangle;
        //detection->elem.second = sph;
        detection->elem.first = sph;
        detection->elem.second = triangle;
        detection->id = sph.getIndex();
        return 1;
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

    return 0; // No intersection: passed all tests for intersections !
}


template<class Sphere>
int DiscreteIntersection::computeIntersection(Sphere& sph1, Sphere& sph2, DetectionOutputVector& contacts)
{
    double r = sph1.r() + sph2.r();
    Vector3 dist = sph2.center() - sph1.center();

    if (dist.norm2() >= r*r)
        return 0;

    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);
    detection->normal = dist;
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->distance = distSph1Sph2 - r;
    detection->elem.first = sph1;
    detection->elem.second = sph2;
    detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

    return 1;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(Sphere& /*sph1*/, Cube& /*cube*/, DetectionOutputVector& /*contacts*/)
{
    //to do
    return 0;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(Sphere& sph1, Ray& ray2, DetectionOutputVector& contacts)
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
    if (dist2 >= (radius1*radius1))
        return 0;

    const double dist = sqrt(dist2);

    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);

    detection->point[1] = ray2Origin + ray2Direction*rayPosInside;
    detection->normal = detection->point[1] - sph1Pos;
    detection->normal /= dist;
    detection->point[0] = sph1Pos + detection->normal * radius1;
    detection->distance = dist - radius1;
    detection->elem.first = sph1;
    detection->elem.second = ray2;
    detection->id = ray2.getIndex();

    return 1;
}

template<class Sphere>
bool DiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Sphere&)
{
    return true;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Sphere& e2, DetectionOutputVector& contacts)
{
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + e2.r();
    const DistanceGrid::Real margin = 0.001f + (DistanceGrid::Real)d0;

    Vector3 p2 = e2.center();
    DistanceGrid::Coord p1;

    if (useXForm)
    {
        p1 = r1.multTranspose(p2-t1);
    }
    else p1 = p2;

    if (!grid1->inBBox( p1, margin )) return 0;
    if (!grid1->inGrid( p1 ))
    {
        std::cerr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<std::endl;
        return 0;
    }

    float d = grid1->interp(p1);
    if (d >= margin) return 0;

    Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
    grad.normalize();

    //p1 -= grad * d; // push p1 back to the surface

    contacts.resize(contacts.size()+1);
    DetectionOutput *detection = &*(contacts.end()-1);

    detection->point[0] = Vector3(p1) - grad * d;
    detection->point[1] = Vector3(p2);
    detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
    detection->distance = d - d0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
