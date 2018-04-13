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
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include "SphericalField.h"

namespace sofa
{

namespace component
{

namespace geometry
{

namespace _sphericalfield_
{

SphericalField::SphericalField()
    : d_inside(initData(&d_inside, false, "inside", "If true the field is oriented inside (resp. outside) the sphere. (default = false)"))
    , d_radiusSphere(initData(&d_radiusSphere, 1.0, "radius", "Radius of Sphere emitting the field. (default = 1)"))
    , d_centerSphere(initData(&d_centerSphere, Vec3d(0.0,0.0,0.0), "center", "Position of the Sphere Surface. (default=0 0 0)" ))
{init();
    }

void SphericalField::init()
{
    m_inside = d_inside.getValue();
    m_center = d_centerSphere.getValue();
    m_radius = d_radiusSphere.getValue();
}

void SphericalField::reinit()
{
    init();
}

double SphericalField::getValue(Vec3d& Pos, int& domain)
{
    SOFA_UNUSED(domain) ;
    double result = (Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
            (Pos[2] - m_center[2])*(Pos[2] - m_center[2]) -
            m_radius * m_radius ;
    if(m_inside)
        result = -result;

    return result;
}

Vec3d SphericalField::getGradient(Vec3d &Pos, int &domain)
{
    SOFA_UNUSED(domain);
    Vec3d g;
    if (m_inside)
    {
        g[0] = -2* (Pos[0] - m_center[0]);
        g[1] = -2* (Pos[1] - m_center[1]);
        g[2] = -2* (Pos[2] - m_center[2]);
    }
    else
    {
        g[0] = 2* (Pos[0] - m_center[0]);
        g[1] = 2* (Pos[1] - m_center[1]);
        g[2] = 2* (Pos[2] - m_center[2]);
    }

    return g;
}

void SphericalField::getValueAndGradient(Vec3d& Pos, double &value, Vec3d& /*grad*/, int& domain)
{
    SOFA_UNUSED(domain);
    Vec3d g;
    g[0] = (Pos[0] - m_center[0]);
    g[1] = (Pos[1] - m_center[1]);
    g[2] = (Pos[2] - m_center[2]);
    if (m_inside)
    {
        value = m_radius*m_radius - g.norm2();
        g = g * (-2);
    }
    else
    {
        value = g.norm2() - m_radius*m_radius;
        g = g * 2;
    }

    return;
}


SOFA_DECL_CLASS(SphericalField)

// Register in the Factory
int SphericalFieldClass = core::RegisterObject("A spherical implicit field.")
        .add< SphericalField >()
        ;

} /// _sphericalfield_
} /// implicit
} /// component
} /// sofa
