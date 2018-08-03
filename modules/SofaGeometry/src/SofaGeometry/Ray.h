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
#ifndef SOFAGEOMETRY_RAY_H
#define SOFAGEOMETRY_RAY_H

#include <sofa/defaulttype/Vec.h>

#include <SofaGeometry/config.h>
#include <SofaGeometry/Constants.h>

namespace sofageometry
{
    /// A ray in 3D.
    /// the ray has an origin and direction vector.
    class Ray
    {
    public:
        Vec3d direction ;
        Vec3d origin ;

        /// Create a new ray starting from 'origin' and pointing to 'direction'
        Ray(const Vec3d& origin=Constants::Origin, const Vec3d& direction=Constants::XAxis);

        /// Return a point along the ray at a given 'p' distance .
        Vec3d getPoint(const double p) const ;
    };
}

#endif // SOFAGEOMETRY_RAY_H
