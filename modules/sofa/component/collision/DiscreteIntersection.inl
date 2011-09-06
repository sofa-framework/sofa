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
#ifndef SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_INL
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/helper/proximity.h>
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

template<class Sphere>
bool DiscreteIntersection::testIntersection(Sphere& sph1, Sphere& sph2)
{
    //sout<<"Collision between Sphere - Sphere"<<sendl;
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
bool DiscreteIntersection::testIntersection(Ray& ray1, Sphere& sph2)
{
    //sout<<"intersectionSphereRay: Collision between Sphere - Ray"<<sendl;
    // Center of the sphere
    const Vector3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vector3 ray1Origin(ray1.origin());
    const Vector3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vector3 tmp = sph2Pos - ray1Origin;
    const double rayPos = tmp*ray1Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

template<class Sphere>
int DiscreteIntersection::computeIntersection( Sphere& sph, Triangle& triangle, OutputVector* contacts)
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
        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);
        detection->normal = -normal;
        detection->point[1] = projPoint;
        detection->point[0] = sph.center() - detection->normal * sph.r();

        detection->value = -distance;
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
int DiscreteIntersection::computeIntersection(Sphere& sph1, Sphere& sph2, OutputVector* contacts)
{
    double r = sph1.r() + sph2.r();
    Vector3 dist = sph2.center() - sph1.center();

    if (dist.norm2() >= r*r)
        return 0;

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->normal = dist;
    double distSph1Sph2 = detection->normal.norm();
    detection->normal /= distSph1Sph2;
    detection->point[0] = sph1.center() + detection->normal * sph1.r();
    detection->point[1] = sph2.center() - detection->normal * sph2.r();

    detection->value = distSph1Sph2 - r;
    detection->elem.first = sph1;
    detection->elem.second = sph2;
    detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

    return 1;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(Sphere& /*sph1*/, Cube& /*cube*/, OutputVector* /*contacts*/)
{
    //to do
    return 0;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(Ray& ray1, Sphere& sph2, OutputVector* contacts)
{
    // Center of the sphere
    const Vector3 sph2Pos(sph2.center());
    // Radius of the sphere
    const double radius1 = sph2.r();

    const Vector3 ray1Origin(ray1.origin());
    const Vector3 ray1Direction(ray1.direction());
    const double length2 = ray1.l();
    const Vector3 tmp = sph2Pos - ray1Origin;
    const double rayPos = tmp*ray1Direction;
    const double rayPosInside = std::max(std::min(rayPos,length2),0.0);
    const double dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    if (dist2 >= (radius1*radius1))
        return 0;

    const double dist = sqrt(dist2);

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0] = ray1Origin + ray1Direction*rayPosInside;
    detection->normal = sph2Pos - detection->point[0];
    detection->normal /= dist;
    detection->point[1] = sph2Pos - detection->normal * radius1;
    detection->value = dist - radius1;
    detection->elem.first = ray1;
    detection->elem.second = sph2;
    detection->id = ray1.getIndex();

    return 1;
}

template<class Sphere>
bool DiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, Sphere&)
{
    return true;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, Sphere& e2, OutputVector* contacts)
{
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const Vector3& t1 = e1.getTranslation();
    const Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + this->getContactDistance() + e2.r();
    const SReal margin = 0.001f + (SReal)d0;

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
        serr << "WARNING: margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName()<<sendl;
        return 0;
    }

    SReal d = grid1->interp(p1);
    if (d >= margin) return 0;

    Vector3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
    grad.normalize();

    //p1 -= grad * d; // push p1 back to the surface

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0] = Vector3(p1) - grad * d;
    detection->point[1] = Vector3(p2);
    detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
    detection->value = d - d0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

template<class Sphere>
bool DiscreteIntersection::testIntersection(FFDDistanceGridCollisionElement&, Sphere&)
{
    return true;
}

template<class Sphere>
int DiscreteIntersection::computeIntersection(FFDDistanceGridCollisionElement& e1, Sphere& e2, OutputVector* contacts)
{

    DistanceGrid* grid1 = e1.getGrid();
    FFDDistanceGridCollisionModel::DeformedCube& c1 = e1.getCollisionModel()->getDeformCube(e1.getIndex());

    const double d0 = e1.getProximity() + e2.getProximity() + getContactDistance() + e2.r();
    const SReal margin = 0.001f + (SReal)d0;

    c1.updateFaces();
    const SReal cubesize = c1.invDP.norm();
    Vector3 p2 = e2.center();
    DistanceGrid::Coord p1 = p2;

    // estimate the barycentric coordinates
    DistanceGrid::Coord b = c1.undeform0(p1);

    // refine the estimate until we are very close to the p2 or we are sure p2 cannot intersect with the object
    int iter;
    SReal err1 = 1000.0f;
    for(iter=0; iter<5; ++iter)
    {
        DistanceGrid::Coord pdeform = c1.deform(b);
        DistanceGrid::Coord diff = p1-pdeform;
        SReal err = diff.norm();
        if (iter>3)
            sout << "Iter"<<iter<<": "<<err1<<" -> "<<err<<" b = "<<b<<" diff = "<<diff<<" d = "<<grid1->interp(c1.initpos(b))<<""<<sendl;
        SReal berr = err*cubesize; if (berr>0.5f) berr=0.5f;
        if (b[0] < -berr || b[0] > 1+berr
            || b[1] < -berr || b[1] > 1+berr
            || b[2] < -berr || b[2] > 1+berr)
            break; // far from the cube
        if (err < 0.005f)
        {
            // we found the corresponding point, but is is only valid if inside the current cube
            if (b[0] > 0.001f && b[0] < 0.999f
                && b[1] > 0.001f && b[1] < 0.999f
                && b[2] > 0.001f && b[2] < 0.999f)
            {
                DistanceGrid::Coord pinit = c1.initpos(b);
                SReal d = grid1->interp(pinit);
                if (d < margin)
                {
                    DistanceGrid::Coord grad = grid1->grad(pinit); // note that there are some redundant computations between interp() and grad()
                    grad.normalize();
                    pinit -= grad*d;
                    grad = c1.deformDir(c1.baryCoords(pinit),grad);
                    grad.normalize();

                    contacts->resize(contacts->size()+1);
                    DetectionOutput *detection = &*(contacts->end()-1);

                    detection->point[0] = Vector3(pinit);
                    detection->point[1] = Vector3(p2);
                    detection->normal = Vector3(grad); // normal in global space from p1's surface
                    detection->value = d - d0;
                    detection->elem.first = e1;
                    detection->elem.second = e2;
                    detection->id = e2.getIndex();
                    return 1;
                }
            }
            break;
        }
        err1 = err;
        SReal d = grid1->interp(c1.initpos(b));
        if (d*0.5f - err > margin)
            break; // the point is too far from the object
        // we are solving for deform(b+db)-deform(b) = p1-deform(b)
        // deform(b+db) ~= deform(b) + M db  -> M db = p1-deform(b) -> db = M^-1 (p1-deform(b))
        b += c1.undeformDir( b, diff );
    }
    if (iter == 5)
    {
        if (b[0] > 0.001f && b[0] < 0.999f
            && b[1] > 0.001f && b[1] < 0.999f
            && b[2] > 0.001f && b[2] < 0.999f)
            serr << "ERROR: FFD-FFD collision failed to converge to undeformed point: p1 = "<<p1<<" b = "<<b<<" c000 = "<<c1.corners[0]<<" c100 = "<<c1.corners[1]<<" c010 = "<<c1.corners[2]<<" c110 = "<<c1.corners[3]<<" c001 = "<<c1.corners[4]<<" c101 = "<<c1.corners[5]<<" c011 = "<<c1.corners[6]<<" c111 = "<<c1.corners[7]<<" pinit = "<<c1.initpos(b)<<" pdeform = "<<c1.deform(b)<<" err = "<<err1<<sendl;
    }

    return 0;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
