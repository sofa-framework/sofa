/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/detection/intersection/RayDiscreteIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <algorithm>


namespace sofa::component::collision::detection::intersection
{


template<class T>
bool RayDiscreteIntersection::testIntersection(collision::geometry::Ray & ray1, collision::geometry::TSphere<T>& sph2)
{
    // Center of the sphere
    const type::Vec3 sph2Pos(sph2.center());
    // Radius of the sphere
    const SReal radius1 = sph2.r();

    const type::Vec3 ray1Origin(ray1.origin());
    const type::Vec3 ray1Direction(ray1.direction());
    const SReal length2 = ray1.l();
    const type::Vec3 tmp = sph2Pos - ray1Origin;
    const SReal rayPos = tmp*ray1Direction;
    const SReal rayPosInside = std::max(std::min(rayPos,length2),(SReal)0.0);
    const SReal dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    return (dist2 < (radius1*radius1));
}

template<class T>
int RayDiscreteIntersection::computeIntersection(collision::geometry::Ray& ray1, collision::geometry::TSphere<T>& sph2, OutputVector* contacts)
{
    // Center of the sphere
    const type::Vec3 sph2Pos(sph2.center());
    // Radius of the sphere
    const SReal radius1 = sph2.r();

    const type::Vec3 ray1Origin(ray1.origin());
    const type::Vec3 ray1Direction(ray1.direction());
    const SReal length2 = ray1.l();
    const type::Vec3 tmp = sph2Pos - ray1Origin;
    const SReal rayPos = tmp*ray1Direction;
    const SReal rayPosInside = std::max(std::min(rayPos,length2),(SReal)0.0);
    const SReal dist2 = tmp.norm2() - (rayPosInside*rayPosInside);
    if (dist2 >= (radius1*radius1))
        return 0;

    const SReal dist = sqrt(dist2);

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

    detection->point[0] = ray1Origin + ray1Direction*rayPosInside;
    detection->normal = sph2Pos - detection->point[0];
    detection->normal /= dist;
    detection->point[1] = sph2.getContactPointByNormal( detection->normal );
    detection->value = dist - radius1;
    detection->elem.first = ray1;
    detection->elem.second = sph2;
    detection->id = ray1.getIndex();

    return 1;
}

} //namespace sofa::component::collision::detection::intersection
