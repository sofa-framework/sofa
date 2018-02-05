/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "ScalarField.h"
namespace sofa
{

namespace component
{

namespace geometry
{

namespace _scalarfield_
{

Vec3d ScalarField::getGradientByFinitDifference(Vec3d& pos, int& i)
{
    double epsilon=0.0001;
    Vec3d Result;
    pos[0] += epsilon;
    Result[0] = getValue(pos, i);
    pos[0] -= epsilon;
    pos[1] += epsilon;
    Result[1] = getValue(pos, i);
    pos[1] -= epsilon;
    pos[2] += epsilon;
    Result[2] = getValue(pos, i);
    pos[2] -= epsilon;

    double v = getValue(pos, i);
    Result[0] = (Result[0]-v)/epsilon;
    Result[1] = (Result[1]-v)/epsilon;
    Result[2] = (Result[2]-v)/epsilon;

    return Result;
}

Vec3d ScalarField::getGradient(Vec3d& pos, int& i)
{
    return getGradientByFinitDifference(pos, i);
}

void ScalarField::getValueAndGradient(Vec3d& pos, double &value, Vec3d& grad, int& domain)
{
  value = getValue(pos,domain);
  grad = getGradient(pos,domain);
}


bool ScalarField::computeSegIntersection(Vec3d& posInside, Vec3d& posOutside, Vec3d& intersecPos, int i)
{


    double tolerance = 0.00001; // tolerance sur la précision m

    float a = (float)getValue(posInside, i);
    float b = (float)getValue(posOutside, i);

    if (a*b>0)
    {
        msg_warning()<<"les deux points sont du même côté de la surface \n";
        return false;
    }

    if(b<0)
    {
        msg_warning()<<"posOutside is inside";
        return false;
    }



    Vec3d Seg = posInside-posOutside;
    if (Seg.norm() < tolerance) // TODO : macro on the global precision
    {
        intersecPos = posOutside;
        return true;
    }

    // we start on posInside and search for the first point outside with a step given by scale //
    int count=0;
    Vec3d step = Seg;
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
            msg_warning()<<": val is negative\n" ;
    }

    if (count>998)
    {
        msg_error()<<"in computeSegIntersection: Seg : "<<Seg;
    }



    return true;





}

void ScalarField::projectPointonSurface(Vec3d& point, int i)
{

    Vec3d grad;
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
        msg_warning() << "No Convergence in projecting the contact point" << msgendl
                      << "-  grad :"	<< grad  ;
    }
}



bool ScalarField::projectPointonSurface2(Vec3d& point, int i, Vec3d& dir)
{


    Vec3d posInside, posOutside;
    if (dir.norm()< 0.001)
    {
        msg_warning() << "grad is too small" ;
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
        dmsg_warning() << "no projection found in ImplSurf::projectPointonSurface(Vec3d& point, Vec3d& dir)";
        return false;
    }
    return computeSegIntersection(posInside, posOutside, point, i);


}


bool ScalarField::projectPointOutOfSurface(Vec3d& point, int i, Vec3d& dir, double &dist_out)
{


    if (projectPointonSurface2(point, i, dir))
    {
        Vec3d grad = getGradient(point, i);
        grad.normalize();
        point += grad*dist_out;
        return true;
    }
    dmsg_warning() << " problem while computing 'projectPointOutOfSurface" ;
    return false;


}

} /// namespace _scalarfield_

} /// namespace geometry

} /// namespace component

} /// namespace sofa
