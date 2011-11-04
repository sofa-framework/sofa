/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_RAYDISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_RAYDISCRETEINTERSECTION_INL
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/RayDiscreteIntersection.h>
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
bool RayDiscreteIntersection::testIntersection(Ray& ray1, Sphere& sph2)
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
int RayDiscreteIntersection::computeIntersection(Ray& ray1, Sphere& sph2, OutputVector* contacts)
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

} // namespace collision

} // namespace component

} // namespace sofa

#endif
