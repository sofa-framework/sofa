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

#include <sofa/type/config.h>
#include <sofa/type/Vec.h>

namespace sofa::type
{

/// Representation of rays.
/// A ray is an infinite line starting at origin and going in some direction.
class Ray
{
public:
    Ray(const Vec3& origin = Vec3(0,0,0), const Vec3& direction = Vec3(0,0,0))
    {
        m_origin = origin;
        m_direction = direction.normalized();
    }

    const Vec3& origin() const { return m_origin; }
    const Vec3& direction() const { return m_direction; }

    Vec3 getPoint(double z) const //< Returns a point at distance units along the ray.
    {
        return m_origin + (m_direction * z);
    }

    void setOrigin(const Vec3& origin) { m_origin = origin; }
    void setDirection(const Vec3& direction) { m_direction = direction.normalized(); }

private:
    Vec3 m_origin;
    Vec3 m_direction;
};

} /// namespace sofa::type
