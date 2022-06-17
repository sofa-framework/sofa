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
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/component/collision/geometry/RayModel.h>
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

    bool testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, geometry::Point&);
    template<class T> bool testIntersection(FFDDistanceGridCollisionElement&, geometry::TSphere<T>&);
    bool testIntersection(FFDDistanceGridCollisionElement&, geometry::Triangle&);
    bool testIntersection(geometry::Ray&, FFDDistanceGridCollisionElement&);

    int computeIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, geometry::Point&, OutputVector*);
    template<class T> int computeIntersection(FFDDistanceGridCollisionElement&, geometry::TSphere<T>&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, geometry::Triangle&, OutputVector*);
    int computeIntersection(geometry::Ray&, FFDDistanceGridCollisionElement&, OutputVector*);

protected:

    detection::intersection::DiscreteIntersection* intersection;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
