/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_TETRAHEDRONDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_TETRAHEDRONDISCRETEINTERSECTION_H
#include "config.h"

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaMiscCollision/TetrahedronModel.h>
#include <SofaBaseCollision/DiscreteIntersection.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_MISC_COLLISION_API TetrahedronDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    TetrahedronDiscreteIntersection(DiscreteIntersection* object);

    bool testIntersection(Tetrahedron&, Point&);
    bool testIntersection(Ray&, Tetrahedron&);

    int computeIntersection(Tetrahedron&, Point&, OutputVector*);
    int computeIntersection(Ray&, Tetrahedron&, OutputVector*);

protected:

    DiscreteIntersection* intersection;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
