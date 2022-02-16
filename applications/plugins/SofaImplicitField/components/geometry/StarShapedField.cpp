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
#include <SofaImplicitField/config.h>
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include "StarShapedField.h"

namespace sofa::component::geometry::_StarShapedField_
{


StarShapedField::StarShapedField()
    : d_inside(initData(&d_inside, false, "inside", "If true the field is oriented inside (resp. outside) the sphere. (default = false)"))
    , d_radiusSphere(initData(&d_radiusSphere, 1.0, "radius", "Radius of Sphere emitting the field. (default = 1)"))
    , d_centerSphere(initData(&d_centerSphere, Vec3d(0.0,0.0,0.0), "center", "Position of the Sphere Surface. (default=0 0 0)" ))
    , d_branches(initData(&d_branches, 1.0, "branches", "Number of branches of the star. (default=1)" ))
    , d_branchesRadius(initData(&d_branchesRadius, 1.0, "branchesRadius", "Size of the branches of the star. (default=1)" ))
{
    init();
    addUpdateCallback("myUpdateCallback", {&d_inside, &d_radiusSphere, &d_centerSphere, &d_branches, &d_branchesRadius}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        this->init();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});

}

void StarShapedField::init()
{
    m_inside = d_inside.getValue();
    m_center = d_centerSphere.getValue();
    m_radius = d_radiusSphere.getValue();
    m_branches = d_branches.getValue();
    m_branchesRadius = d_branchesRadius.getValue();
}

void StarShapedField::reinit()
{
    init();
}

double StarShapedField::getValue(Vec3d& Pos, int& domain)
{
    SOFA_UNUSED(domain) ;
    double length = sqrt((Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
            (Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
    double result = length - m_radius ;

    double simpleCos0 = (Pos[0] - m_center[0]);
    double simpleCos1 = (Pos[1] - m_center[1]);
    double simpleCos2 = (Pos[2] - m_center[2]);

    result += m_branchesRadius * (  cos( m_branches*simpleCos0 ) + cos( m_branches*simpleCos1 ) +   cos( m_branches*simpleCos2 ) );
    if(m_inside)
        result = -result;

    return result;
}

Vec3d StarShapedField::getGradient(Vec3d &Pos, int &domain)
{
    SOFA_UNUSED(domain);
    Vec3d g;

    double length = sqrt((Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
            (Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
    double deltaLength0 = (Pos[0] - m_center[0])/length;
    double deltaLength1 = (Pos[1] - m_center[1])/length;
    double deltaLength2 = (Pos[2] - m_center[2])/length;

    g[0] = deltaLength0 - m_branchesRadius*m_branches*sin(m_branches*(Pos[0] - m_center[0]));
    g[1] = deltaLength1 - m_branchesRadius*m_branches*sin(m_branches*(Pos[1] - m_center[1]));
    g[2] = deltaLength2 - m_branchesRadius*m_branches*sin(m_branches*(Pos[2] - m_center[2]));

    if (m_inside)
    {
        g[0] = -g[0];
        g[1] = -g[1];
        g[2] = -g[2];
    }

    return g;
}

void StarShapedField::getHessian(Vec3d &Pos, Mat3x3& h)
{
    double length = sqrt((Pos[0] - m_center[0])*(Pos[0] - m_center[0]) +
            (Pos[1] - m_center[1])*(Pos[1] - m_center[1]) +
            (Pos[2] - m_center[2])*(Pos[2] - m_center[2]));
    double deltaLength0 = (Pos[0] - m_center[0])/length;
    double deltaLength1 = (Pos[1] - m_center[1])/length;
    double deltaLength2 = (Pos[2] - m_center[2])/length;


    h[0][0] = (length - (Pos[0] - m_center[0])*deltaLength0 )/(length*length) - m_branchesRadius*m_branches*m_branches*cos(m_branches*(Pos[0] - m_center[0]));
    h[1][1] = (length - (Pos[1] - m_center[1])*deltaLength1 )/(length*length) - m_branchesRadius*m_branches*m_branches*cos(m_branches*(Pos[1] - m_center[1]));
    h[2][2] = (length - (Pos[2] - m_center[2])*deltaLength2 )/(length*length) - m_branchesRadius*m_branches*m_branches*cos(m_branches*(Pos[2] - m_center[2]));

    h[0][1] = h[1][0] = - (Pos[0] - m_center[0])*(Pos[1] - m_center[1]) / (length*length*length);
    h[0][2] = h[2][0] = - (Pos[0] - m_center[0])*(Pos[2] - m_center[2]) / (length*length*length);
    h[1][2] = h[2][1] = - (Pos[2] - m_center[2])*(Pos[1] - m_center[1]) / (length*length*length);

    return;
}

// Register in the Factory
int StarShapedFieldClass = core::RegisterObject("A spherical implicit field.")
        .add< StarShapedField >()
        ;

} // namespace sofa::component::geometry::_StarShapedField_

