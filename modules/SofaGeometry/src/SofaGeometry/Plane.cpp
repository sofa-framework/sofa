/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*****************************************************************************
 * Contributors:
 *     - damien.marchal@univ-lille1.fr
 ****************************************************************************/
#include <Eigen/Geometry>
#include <Eigen/Core>
using Eigen::ParametrizedLine ;
using Eigen::Hyperplane ;
using Eigen::Vector3d ;

#define toEigen(v) Vector3d(v.x(), v.y(), v.z())

#include <SofaGeometry/Ray.h>
#include <SofaGeometry/Plane.h>
namespace sofageometry
{

// Create a new plan object by passing the hessian-form of the place
// see: http://mathworld.wolfram.com/Plane.html
// the normal parameter must be a normalized vector.
// the distance parameter is the distance to the origin.
Plane::Plane(const Vec3d& normal, const double& distance){
    this->normal = normal ;
    this->distance = distance ;
}

Plane::Plane(const Vec3d& normal, const Vec3d& point)
{
    double norm = normal.norm();
    this->normal = normal/norm ;

    // Conversion is taken from:
    // http://mathworld.wolfram.com/Plane.html
    double d =     - (normal.x()*point.x())
                   - (normal.y()*point.y())
                   - (normal.z()*point.z()) ;

    this->distance = d / norm ;
}

// Casts a rayon against the plane, returning true or false on
// intersection and, if true the intesersection position.
bool Plane::raycast(const Ray& ray, double& p) const {
    Hyperplane<double, 3> plane(toEigen(normal), distance) ;

    ParametrizedLine<double, 3> line(toEigen(ray.origin),
                                     toEigen(ray.direction)) ;

    p = line.intersectionParameter(plane) ;
    return p >= 0 ;
}

}
