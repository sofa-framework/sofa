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
#ifndef SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_H
#include <SofaDistanceGrid/config.h>

#include <sofa/core/collision/Intersection.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/LineCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/component/collision/geometry/RayCollisionModel.h>
#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>

#include <SofaDistanceGrid/components/collision/DistanceGridCollisionModel.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_SOFADISTANCEGRID_API FFDDistanceGridDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef detection::intersection::DiscreteIntersection::OutputVector OutputVector;

public:
    FFDDistanceGridDiscreteIntersection(detection::intersection::DiscreteIntersection* object);

    bool testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, const core::collision::Intersection*);
    bool testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, const core::collision::Intersection*);
    bool testIntersection(FFDDistanceGridCollisionElement&, geometry::Point&, const core::collision::Intersection*);
    template<class T> bool testIntersection(FFDDistanceGridCollisionElement&, geometry::TSphere<T>&, const core::collision::Intersection*);
    bool testIntersection(FFDDistanceGridCollisionElement&, geometry::Triangle&, const core::collision::Intersection*);
    bool testIntersection(geometry::Ray&, FFDDistanceGridCollisionElement&, const core::collision::Intersection*);

    int computeIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(FFDDistanceGridCollisionElement&, geometry::Point&, OutputVector*, const core::collision::Intersection*);
    template<class T> int computeIntersection(FFDDistanceGridCollisionElement&, geometry::TSphere<T>&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(FFDDistanceGridCollisionElement&, geometry::Triangle&, OutputVector*, const core::collision::Intersection*);
    int computeIntersection(geometry::Ray&, FFDDistanceGridCollisionElement&, OutputVector*, const core::collision::Intersection*);

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
