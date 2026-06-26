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
#ifndef SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_H
#include <SofaDistanceGrid/config.h>

#include <sofa/core/collision/Intersection.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/LineCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/component/collision/geometry/RayCollisionModel.h>
#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>

#include "DistanceGridCollisionModel.h"

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_SOFADISTANCEGRID_API RigidDistanceGridDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef detection::intersection::DiscreteIntersection::OutputVector OutputVector;

public:
    RigidDistanceGridDiscreteIntersection(detection::intersection::DiscreteIntersection* intersection);

    bool testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, const core::collision::Intersection* );
    bool testIntersection(RigidDistanceGridCollisionElement&, geometry::Point&, const core::collision::Intersection* );
    template<class T> bool testIntersection(RigidDistanceGridCollisionElement&, geometry::TSphere<T>&, const core::collision::Intersection* );
    bool testIntersection(RigidDistanceGridCollisionElement&, geometry::Line&, const core::collision::Intersection* );
    bool testIntersection(RigidDistanceGridCollisionElement&, geometry::Triangle&, const core::collision::Intersection* );
    bool testIntersection(geometry::Ray&, RigidDistanceGridCollisionElement&, const core::collision::Intersection*);

    int computeIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(RigidDistanceGridCollisionElement&, geometry::Point&, OutputVector*, const core::collision::Intersection*);
    template<class T> int computeIntersection(RigidDistanceGridCollisionElement&, geometry::TSphere<T>&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(RigidDistanceGridCollisionElement&, geometry::Line&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(RigidDistanceGridCollisionElement&, geometry::Triangle&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(geometry::Ray&, RigidDistanceGridCollisionElement&, OutputVector*, const core::collision::Intersection*);

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
