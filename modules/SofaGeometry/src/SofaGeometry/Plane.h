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
#ifndef SOFAGEOMETRY_PLANE_H
#define SOFAGEOMETRY_PLANE_H

#include <sofa/defaulttype/Vec3Types.h>

#include <SofaGeometry/config.h>
#include <SofaGeometry/Constants.h>

////////////////////////////// FORWARD DECLARATION /////////////////////////////////////////////////
namespace sofageometry { class Ray; }

/////////////////////////////////// DECLARATION ////////////////////////////////////////////////////
namespace sofageometry
{

/// A Plane in 3D.
/// the plane orientation is represented by the normal vector and the distance to the origin
/// along this line vector.
class SOFAGEOMETRY_API Plane
{
public:
    Vec3d   normal ;   /// normal vector of the plane
    double  distance;  /// distance to the origin

    /// Create a new plan object.
    /// the 'normal' parameter must be a normalized vector.
    /// the 'distance' parameter is the distance of the plane to the origin.
    Plane(const Vec3d& normal=Constants::XAxis, const double& distance=0) ;

    /// Create a new plan object.
    /// the 'normal' parameter must be a normalized vector.
    /// the 'point' parameter is a point by which the plane is passing through.
    Plane(const Vec3d& normal, const Vec3d& point) ;

    /// Casts a ray against the plane, returning true or false on
    /// intersection and, if true the intesersection position.
    bool raycast(const Ray& ray, double& p) const ;
};

}

#endif /// SOFAGEOMETRY_PLANE_H
