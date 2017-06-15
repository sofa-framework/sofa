/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_MESHDISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_MESHDISCRETEINTERSECTION_INL
#include <sofa/helper/system/config.h>
#include <SofaGeneralMeshCollision/MeshDiscreteIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{


template<class T>
bool MeshDiscreteIntersection::testIntersection( TSphere<T>& sph, Triangle& triangle)
{
    double EPSILON = 0.00001;
    //Vertices of the triangle:
    defaulttype::Vector3 p0 = triangle.p1();
    defaulttype::Vector3 p1 = triangle.p2();
    defaulttype::Vector3 p2 = triangle.p3();

    // Center of the sphere
    const defaulttype::Vector3 sphCenter(sph.center());
    // Radius of the sphere
    const double r = sph.r();

    //Normal to the plane (plane spanned by tree points of the triangle)
    defaulttype::Vector3 normal = cross( (p1 - p0), (p2 - p0) );
    normal.normalize();

    //Distance from the center of the sphere to the plane.
    double distance = sphCenter*normal - normal*p0;

    //Projection of the center of the sphere onto the plane
    defaulttype::Vector3 projPoint = sphCenter - normal*distance;

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

template<class T>
int MeshDiscreteIntersection::computeIntersection( TSphere<T>& sph, Triangle& triangle, OutputVector* contacts)
{
    double EPSILON = 0.00001;
    //Vertices of the triangle:
    defaulttype::Vector3 p0 = triangle.p1();
    defaulttype::Vector3 p1 = triangle.p2();
    defaulttype::Vector3 p2 = triangle.p3();

    // Center of the sphere
    const defaulttype::Vector3 sphCenter(sph.center());
    // Radius of the sphere
    const double r = sph.r();

    //Normal to the plane (plane spanned by tree points of the triangle)
    defaulttype::Vector3 normal = cross( (p1 - p0), (p2 - p0) );
    normal.normalize();

    //Distance from the center of the sphere to the plane.
    double distance = sphCenter*normal - normal*p0;

    //Projection of the center of the sphere onto the plane
    defaulttype::Vector3 projPoint = sphCenter - normal*distance;

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
        core::collision::DetectionOutput *detection = &*(contacts->end()-1);
        detection->normal = -normal;
        detection->point[1] = projPoint;
        detection->point[0] = sph.getContactPointByNormal( detection->normal );
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


inline int MeshDiscreteIntersection::computeIntersection(Capsule & cap,Triangle & tri,OutputVector* contacts){
    return MeshIntTool::computeIntersection(cap,tri,intersection->getAlarmDistance(),intersection->getContactDistance(),contacts);
}

inline int MeshDiscreteIntersection::computeIntersection(Capsule & cap,Line & lin,OutputVector* contacts){
    return MeshIntTool::computeIntersection(cap,lin,intersection->getAlarmDistance(),intersection->getContactDistance(),contacts);
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
