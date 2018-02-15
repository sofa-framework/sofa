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
#ifndef SOFA_COMPONENT_RAYTRIANGLEINTERSECTION_H
#define SOFA_COMPONENT_RAYTRIANGLEINTERSECTION_H
#include "config.h"

#include <SofaMeshCollision/Triangle.h>

namespace sofa
{

namespace component
{

namespace collision
{


/// this class computes if a Triangle P intersects a line segment

class SOFA_MESH_COLLISION_API RayTriangleIntersection
{
public:
    RayTriangleIntersection(); // start a Proximity solver
    ~RayTriangleIntersection();

    bool NewComputation( const sofa::defaulttype::Vector3 &p1, const sofa::defaulttype::Vector3 &p2, const sofa::defaulttype::Vector3 &p3, const sofa::defaulttype::Vector3 &origin, const sofa::defaulttype::Vector3 &direction,  SReal &t,  SReal &u, SReal &v);
    bool NewComputation( Triangle *triP, const sofa::defaulttype::Vector3 &origin, const sofa::defaulttype::Vector3 &direction,  SReal &t,  SReal &u, SReal &v)
    {
        return NewComputation( triP->p1(), triP->p2(), triP->p3(), origin, direction, t, u, v);
    }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
