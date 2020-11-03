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
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include "BottleField.h"

namespace sofa::component::geometry::_BottleField_
{

using sofa::defaulttype::Vec2;

BottleField::BottleField()
    : d_inside(initData(&d_inside, false, "inside", "If true the field is oriented inside (resp. outside) the bottle-shaped object. (default = false)"))
    , d_radiusSphere(initData(&d_radiusSphere, 1.0, "radius", "Radius of Sphere emitting the field. (default = 1)"))
    , d_centerSphere(initData(&d_centerSphere, Vec3d(0.0,0.0,0.0), "center", "Position of the Sphere Surface. (default=0 0 0)" ))
    , d_shift(initData(&d_shift, 1.0, "shift", "How much the top ellipsoid is shifted from the bottom sphere. (default=1)" ))
    , d_ellipsoidRadius(initData(&d_ellipsoidRadius, 1.0, "ellipsoidRadius", "Radius of the ellipsoid whose intersection with the sphere is taken off" ))
    , d_excentricity(initData(&d_excentricity, 1.0, "excentricity", "excentricity of ellipsoid" ))
{
    init();
    addUpdateCallback("myUpdateCallback", {&d_inside, &d_radiusSphere, &d_centerSphere, &d_shift, &d_ellipsoidRadius, &d_excentricity}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        this->init();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}

void BottleField::init()
{
    m_inside = d_inside.getValue();
    m_center = d_centerSphere.getValue();
    m_radius = d_radiusSphere.getValue();
    m_shift = d_shift.getValue();
    m_ellipsoidRadius = d_ellipsoidRadius.getValue();
    m_excentricity = d_excentricity.getValue();
}

void BottleField::reinit()
{
    init();
}

double BottleField::outerLength(Vec3d& Pos)
{
    return sqrt((Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
                (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
                (Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
}

double BottleField::innerLength(Vec3d& Pos)
{
    return sqrt(m_excentricity*(Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - (m_center[1]+m_shift))*(Pos[1] - (m_center[1]+m_shift)) +
            m_excentricity*(Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
}

double BottleField::getValue(Vec3d& Pos, int& domain)
{
    SOFA_UNUSED(domain) ;
    double resultSphereOuter = this->outerLength(Pos) - m_radius ;
    double resultEllipsoidInner = this->innerLength(Pos) - m_ellipsoidRadius;

    double result = std::max(resultSphereOuter,-resultEllipsoidInner);

    if(m_inside)
        result = -result;

    return result;
}

Vec3d BottleField::getGradient(Vec3d &Pos, int &domain)
{
    SOFA_UNUSED(domain);
    Vec3d g;

    double LsphereOuter = this->outerLength(Pos)  ;
    double LEllipsoidInner = this->innerLength(Pos) ;

    if (LsphereOuter - m_radius > - (LEllipsoidInner- m_ellipsoidRadius)){
        g[0] = (Pos[0] - m_center[0])/LsphereOuter;
        g[1] = (Pos[1] - m_center[1])/LsphereOuter;
        g[2] = (Pos[2] - m_center[2])/LsphereOuter;
    }
    else
    {
        g[0] = -m_excentricity*(Pos[0] - m_center[0])/LEllipsoidInner;
        g[1] = -(Pos[1] - (m_center[1]+m_shift))/LEllipsoidInner;
        g[2] = -m_excentricity*(Pos[2] - m_center[2])/LEllipsoidInner;
    }


    if (m_inside)
    {
        g[0] = -g[0];
        g[1] = -g[1];
        g[2] = -g[2];
    }

    return g;
}

void BottleField::getHessian(Vec3d &Pos, Mat3x3& h)
{
    double LsphereOuter = this->outerLength(Pos)  ;
    double LEllipsoidInner = this->innerLength(Pos) ;

    if (LsphereOuter - m_radius > - (LEllipsoidInner- m_ellipsoidRadius))
    {
        double LsphereOuterSquare = LsphereOuter*LsphereOuter;
        double LsphereOuterCube = LsphereOuter*LsphereOuter*LsphereOuter;
        h[0][0] = ( LsphereOuter - (Pos[0] - m_center[0])*(Pos[0] - m_center[0])/LsphereOuter )/LsphereOuterSquare ;
        h[1][1] = ( LsphereOuter - (Pos[1] - m_center[1])*(Pos[1] - m_center[1])/LsphereOuter )/LsphereOuterSquare ;
        h[2][2] = ( LsphereOuter - (Pos[2] - m_center[2])*(Pos[2] - m_center[2])/LsphereOuter )/LsphereOuterSquare ;

        h[0][1] = h[1][0] = - (Pos[0] - m_center[0])*(Pos[1] - m_center[1]) / LsphereOuterCube;
        h[0][2] = h[2][0] = - (Pos[0] - m_center[0])*(Pos[2] - m_center[2]) / LsphereOuterCube;
        h[1][2] = h[2][1] = - (Pos[2] - m_center[2])*(Pos[1] - m_center[1]) / LsphereOuterCube;
    }
    else
    {
        double LEllipsoidInnerSquare = LEllipsoidInner*LEllipsoidInner;
        double LEllipsoidInnerCube = LEllipsoidInner*LEllipsoidInner*LEllipsoidInner;
        h[0][0] = -m_excentricity*(LEllipsoidInner - m_excentricity*(Pos[0] - m_center[0])*(Pos[0] - m_center[0])/LEllipsoidInner )/LEllipsoidInnerSquare ;
        h[1][1] = -(LEllipsoidInner - (Pos[1] - (m_center[1]+m_shift))*(Pos[1] - (m_center[1]+m_shift))/LEllipsoidInner )/LEllipsoidInnerSquare ;
        h[2][2] = -m_excentricity*(LEllipsoidInner - m_excentricity*(Pos[2] - m_center[2])*(Pos[2] - m_center[2])/LEllipsoidInner )/LEllipsoidInnerSquare ;

        h[0][1] = h[1][0] = m_excentricity*(Pos[0] - m_center[0])*(Pos[1] - (m_center[1]+m_shift)) / LEllipsoidInnerCube;
        h[0][2] = h[2][0] = m_excentricity*m_excentricity*(Pos[0] - m_center[0])*(Pos[2] - m_center[2]) / LEllipsoidInnerCube;
        h[1][2] = h[2][1] = m_excentricity*(Pos[2] - m_center[2])*(Pos[1] - (m_center[1]+m_shift)) / LEllipsoidInnerCube;
    }

    if (m_inside)
    {
        for (unsigned int i=0; i<3; i++)
            for (unsigned int j=0; j<3; j++)
                h[i][j] = -h[i][j];
    }

    return;
}


// Register in the Factory
int BottleFieldClass = core::RegisterObject("A spherical implicit field.")
        .add< BottleField >()
        ;

} // namespace sofa::component::geometry::_BottleField_
