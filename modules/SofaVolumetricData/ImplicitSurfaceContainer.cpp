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

defaulttype::Vec3d ImplicitSurface::getGradient(defaulttype::Vec3d& Pos, int& i)
{

    double epsilon=0.0001;
    defaulttype::Vec3d Result;
    Pos[0] += epsilon;
    Result[0] = getValue(Pos, i);
    Pos[0] -= epsilon;
    Pos[1] += epsilon;
    Result[1] = getValue(Pos, i);
    Pos[1] -= epsilon;
    Pos[2] += epsilon;
    Result[2] = getValue(Pos, i);
    Pos[2] -= epsilon;

    double v = getValue(Pos, i);
    Result[0] = (Result[0]-v)/epsilon;
    Result[1] = (Result[1]-v)/epsilon;
    Result[2] = (Result[2]-v)/epsilon;

    return Result;
}

bool ImplicitSurface::computeSegIntersection(defaulttype::Vec3d& posInside, defaulttype::Vec3d& posOutside, defaulttype::Vec3d& intersecPos, int i)
{


    double tolerance = 0.00001; // tolerance sur la précision m

    float a = (float)getValue(posInside, i);
    float b = (float)getValue(posOutside, i);

    if (a*b>0)
    {
        std::cerr<<"WARNING : les deux points sont du même côté de la surface \n"<<std::endl;
        return false;
    }

    if(b<0)
    {
        std::cerr<<"WARNING : posOutside is inside"<<std::endl;
        return false;
    }



    defaulttype::Vec3d Seg = posInside-posOutside;
    if (Seg.norm() < tolerance) // TODO : macro on the global precision
    {
        intersecPos = posOutside;
        return true;
    }

    // we start on posInside and search for the first point outside with a step given by scale //
    int count=0;
    defaulttype::Vec3d step = Seg;
    double val = b;
    intersecPos = posOutside;

    double step_incr=0.1;

    while(step.norm()> tolerance && count < 1000)
    {
        step *= step_incr;

        while (val >= 0 && count < (1/step_incr + 1))
        {
            count++;
            intersecPos += step;
            val = getValue(intersecPos, i);
        }

        // we restart with more precision
        intersecPos -=step;

        val = getValue(intersecPos, i);
        if (val < 0)
            std::cerr<<"WARNING : val is negative\n"<<std::endl;

    }

    if (count>998)
    {
        std::cerr<<"ERROR in computeSegIntersection: Seg : "<<Seg<<std::endl;


    }



    return true;





}

void ImplicitSurface::projectPointonSurface(defaulttype::Vec3d& point, int i)
{

    defaulttype::Vec3d grad;
    double value= getValue(point, i);
    int it;
    for (it=0; it<10; it++)
    {
        if (value*value < 0.0000000001)
            break;

        grad = getGradient(point, i);
        point -= grad * (value / dot(grad,grad) );
        value = getValue(point, i);


    }

    if (it==10)
    {
        std::cout<<"No Convergence in projecting the contact point"<<std::endl;
        std::cout<<"-  grad :"	<< grad <<std::endl;
    }
}



bool ImplicitSurface::projectPointonSurface2(defaulttype::Vec3d& point, int i, defaulttype::Vec3d& dir)
{


    defaulttype::Vec3d posInside, posOutside;
    if (dir.norm()< 0.001)
    {
        printf("Warning : grad is null \n");
        return false;
    }

    dir.normalize();
    double value= getValue(point, i);

    double step=0.1;

    int count = 0;
    if(value>0)
    {
        posInside = point;
        while(value>0 && count < 30)
        {
            count++;
            posInside -= dir * step;
            value = getValue(posInside, i);
        }
        posOutside = point;
    }
    else
    {
        posOutside = point;
        while(value<0 && count < 30)
        {
            count++;
            posOutside += dir * step;
            value = getValue(posOutside, i);
        }
        posInside = point;

    }
    if (count == 30)
    {
        printf("\n WARNING : no projection found in  ImplSurf::projectPointonSurface(Vec3d& point, Vec3d& dir)");
        return false;
    }
    return computeSegIntersection(posInside, posOutside, point, i);


}


bool ImplicitSurface::projectPointOutOfSurface(defaulttype::Vec3d& point, int i, defaulttype::Vec3d& dir, double &dist_out)
{


    if (projectPointonSurface2(point, i, dir))
    {
        defaulttype::Vec3d grad = getGradient(point, i);
        grad.normalize();
        point += grad*dist_out;
        return true;
    }
    printf(" pb in projectPointOutOfSurface \n");
    return false;


}



  double SphereSurface::getValue(defaulttype::Vec3d& Pos, int& domain)
{
    //std::cout<<"getValue is called for pos"<<Pos<<"radius="<<_radius <<std::endl;
  (void)domain;
  double result = (Pos[0] - _Center[0])*(Pos[0] - _Center[0]) +
    (Pos[1] - _Center[1])*(Pos[1] - _Center[1]) +
    (Pos[2] - _Center[2])*(Pos[2] - _Center[2]) -
    _radius * _radius ;
  if(_inside)
    result = -result;
  
  return result;
}

defaulttype::Vec3d SphereSurface::getGradient(defaulttype::Vec3d &Pos, int &domain)
{
  (void)domain;
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

void SphereSurface::getValueAndGradient(defaulttype::Vec3d& Pos, double &value, defaulttype::Vec3d& /*grad*/, int& domain)
{
  (void)domain;
  defaulttype::Vec3d g;
  g[0] = (Pos[0] - _Center[0]);
  g[1] = (Pos[1] - _Center[1]);
  g[2] = (Pos[2] - _Center[2]);
  if (_inside)
    {
      value = _radius*_radius - g.norm2();
      g = g * (-2);
    }
  else
    {
      value = g.norm2() - _radius*_radius;
      g = g * 2;
    }
  
  return;
}

double SphereSurface::getDistance(defaulttype::Vec3d& Pos, int& /*domain*/)
{
  double result = _radius - sqrt((Pos[0] - _Center[0])*(Pos[0] - _Center[0]) +
				 (Pos[1] - _Center[1])*(Pos[1] - _Center[1]) +
				 (Pos[2] - _Center[2])*(Pos[2] - _Center[2]));
  return _inside ? result : -result;
}

double SphereSurface::getDistance(defaulttype::Vec3d& /*Pos*/, double value, double grad_norm, int &domain)
{
  (void)domain;
  if (grad_norm < 0) // use value
    grad_norm = sqrt(_inside ? _radius*_radius - value : value + _radius*_radius);
  else grad_norm /= 2;
  return _inside ? _radius - grad_norm : grad_norm - _radius;
}



} // namespace container

} // namespace component

} // namespace sofa
