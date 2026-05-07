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
#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/core/collision/Intersection.h>

#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/RayCollisionModel.h>
#include <sofa/component/collision/geometry/TetrahedronCollisionModel.h>
#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>

namespace sofa::component::collision::detection::intersection
{
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API TetrahedronDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    TetrahedronDiscreteIntersection(DiscreteIntersection* intersection);
    
    bool testIntersection(collision::geometry::Tetrahedron&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Ray&, collision::geometry::Tetrahedron&, const core::collision::Intersection* currentIntersection);

    int computeIntersection(collision::geometry::Tetrahedron&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Ray&, collision::geometry::Tetrahedron&, OutputVector*, const core::collision::Intersection* currentIntersection);
};

} // namespace sofa::component::collision::detection::intersection
