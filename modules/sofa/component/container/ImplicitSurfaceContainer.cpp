/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/core/ObjectFactory.h>
#include "ImplicitSurfaceContainer.h"
namespace sofa
{

namespace component
{

namespace container
{


SOFA_DECL_CLASS(SphereSurface)

// Register in the Factory
int SphereSurfaceClass = core::RegisterObject("")
        .add< SphereSurface >()
        ;


bool ImplicitSurface::computeSegIntersection(defaulttype::Vec3d& posInside, defaulttype::Vec3d& posOutside, defaulttype::Vec3d& intersecPos)
{

    intersecPos = posInside*0.5 + posOutside*0.5;

    defaulttype::Vec3d s = posOutside - posInside;
    int it;
    for (it=0; it<10; it++)						// TODO : mettre le epsilon en parametre
    {
        double value = getValue(intersecPos);


        if(value*value<0.00000001)
        {
            break;
        }
        defaulttype::Vec3d grad = getGradient(intersecPos);
        double a;
        if(fabs( dot (grad,s) ) < 0.00000001 * fabs(value) )
        {
            std::cout<<"++++WARNING:++++++  in computeSegIntersection  dot (grad,s) =" << dot (grad,s) <<std::endl;
            a=0.1; // valeur mise au hasard
        }
        else
            a = value/( dot (grad,s) );



        intersecPos -= s*a;
        if (this->f_printLog.getValue())
        {
            std::cout<<"it "<< it<<" ---  value:"<< value;
            std::cout<<" grad ="<< grad <<" s"<< s;
            std::cout<<" correction ="<<a<<"intersecPos"<<intersecPos<< std::endl;
        }

    }
    if(it==10)
        std::cout<<"++++WARNING:++++++  no convergence in computeSegIntersection : posIn=" << posInside <<" -- posOut="<< posInside<< std::endl;


    return true;

}

void ImplicitSurface::projectPointonSurface(defaulttype::Vec3d& point)
{

    defaulttype::Vec3d grad;
    double value= getValue(point);
    int it;
    for (it=0; it<10; it++)
    {
        grad = getGradient(point);
        point -= grad * (value / dot(grad,grad) );
        value = getValue(point);

        if (value*value < 0.0000000001)
            break;
    }

    if (it==10)
        std::cout<<"No Convergence in projecting the contact point"<<std::endl;
    std::cout<<"-  grad :"	<< grad <<std::endl;
}


double SphereSurface::getValue(defaulttype::Vec3d& Pos)
{
    //std::cout<<"getValue is called for pos"<<Pos<<"radius="<<_radius <<std::endl;

    double result = (Pos[0] - _Center[0])*(Pos[0] - _Center[0]) +
            (Pos[1] - _Center[1])*(Pos[1] - _Center[1]) +
            (Pos[2] - _Center[2])*(Pos[2] - _Center[2]) -
            _radius * _radius ;
    if(_inside)
        result = -result;

    return result;
}

defaulttype::Vec3d SphereSurface::getGradient(defaulttype::Vec3d &Pos)
{
    defaulttype::Vec3d g;
    if (_inside)
    {
        g[0] = -2* (Pos[0] - _Center[0]);
        g[1] = -2* (Pos[1] - _Center[1]);
        g[2] = -2* (Pos[2] - _Center[2]);
    }
    else
    {
        g[0] = 2* (Pos[0] - _Center[0]);
        g[1] = 2* (Pos[1] - _Center[1]);
        g[2] = 2* (Pos[2] - _Center[2]);
    }

    return g;
}






} // namespace container

} // namespace component

} // namespace sofa
