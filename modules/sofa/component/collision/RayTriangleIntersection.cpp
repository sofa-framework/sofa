/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include "RayTriangleIntersection.h"
#include <sofa/helper/LCPSolver.inl>

namespace sofa
{

namespace component
{

namespace collision
{

RayTriangleIntersection::RayTriangleIntersection()
{
}
RayTriangleIntersection::~RayTriangleIntersection()
{
}
bool
RayTriangleIntersection::NewComputation(Triangle *triP, const Vector3 &origin, const Vector3 &direction,   double &t,  double &u, double &v)

{
    double EPSILON = 0.000001;
    t = 0; u = 0; v = 0;

    Vector3 edge1 = triP->p2() - triP->p1();
    Vector3 edge2 = triP->p3() - triP->p1();

    Vector3 tvec, pvec, qvec;
    double det, inv_det;

    pvec = direction.cross(edge2);

    det = dot(edge1, pvec);
    if(det==0.0)
    {

        return false;

    }
// 	    if (fabs(triP->n()*direction) < 0.0000)
//                 return false;

    inv_det = 1.0 / det;

    tvec = origin - triP->p1();

    u = dot(tvec, pvec) * inv_det;
    if (u < 0.000001||u >0.999999900000)
        return false;

    qvec = tvec.cross(edge1);

    v = dot(direction, qvec) * inv_det;
    if (v < 0.00001|| (u +v) > 0.999999900000)
        return false;

    t = dot(edge2, qvec) * inv_det;

    if (t <=0.000001||isnan(t)||isnan(v)||isnan(u))
        return false;

    return true;
}

}
}
}

