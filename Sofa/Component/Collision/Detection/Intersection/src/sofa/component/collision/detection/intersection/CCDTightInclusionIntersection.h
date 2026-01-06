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

#include <sofa/component/collision/detection/intersection/BaseProximityIntersection.h>

#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/component/collision/geometry/RayModel.h>

namespace sofa::component::collision::detection::intersection
{

/**
 * Intersection methods using proximities. Filters are added to limit the number of contacts.
 * The following pairs of collision models are supported:
 * - Cube/Cube
 * - Line/Line
 * - Triangle/Point
 * The following pairs of collision models are ignored:
 * - Sphere/Sphere
 * - Sphere/Point
 * - Point/Point
 * - Line/Point
 * - Line/Sphere
 * - Triangle/Line
 * - Triangle/Triangle
 * - Triangle/Sphere
 * - Ray/Triangle
 * - Ray/Sphere
 * - Ray/Point
 * - Ray/Line
 */
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API CCDTightInclusionIntersection
    : public BaseProximityIntersection
{
   public:
    SOFA_CLASS(CCDTightInclusionIntersection, BaseProximityIntersection);

    typedef core::collision::IntersectorFactory<CCDTightInclusionIntersection> IntersectorFactory;

   protected:
    CCDTightInclusionIntersection();

   public:
    void init() override;

    virtual bool useContinuous() const override;
    virtual core::CollisionModel::ContinuousIntersectionTypeFlag continuousIntersectionType() const;


    bool testIntersection(collision::geometry::Cube&, collision::geometry::Cube&,
                          const core::collision::Intersection* currentIntersection) override;

    // bool testIntersection(collision::geometry::Point&, collision::geometry::Point&, const
    // core::collision::Intersection* currentIntersection); bool
    // testIntersection(collision::geometry::Sphere&, collision::geometry::Point&, const
    // core::collision::Intersection* currentIntersection); bool
    // testIntersection(collision::geometry::Sphere&, collision::geometry::Sphere&, const
    // core::collision::Intersection* currentIntersection) override; bool
    // testIntersection(collision::geometry::Line&, collision::geometry::Point&, const
    // core::collision::Intersection* currentIntersection); bool
    // testIntersection(collision::geometry::Line&, collision::geometry::Sphere&, const
    // core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Line&,
                          const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Point&,
                          const core::collision::Intersection* currentIntersection);
    // int testIntersection(collision::geometry::Triangle&, collision::geometry::Sphere&, const
    // core::collision::Intersection* currentIntersection);

    int computeIntersection(collision::geometry::Cube&, collision::geometry::Cube&, OutputVector*,
                            const core::collision::Intersection* currentIntersection) override;
    // int computeIntersection(collision::geometry::Point&, collision::geometry::Point&,
    // OutputVector*, const core::collision::Intersection* currentIntersection); int
    // computeIntersection(collision::geometry::Sphere&, collision::geometry::Point&, OutputVector*,
    // const core::collision::Intersection* currentIntersection); int
    // computeIntersection(collision::geometry::Sphere&, collision::geometry::Sphere&,
    // OutputVector*, const core::collision::Intersection* currentIntersection) override; int
    // computeIntersection(collision::geometry::Line&, collision::geometry::Point&, OutputVector*,
    // const core::collision::Intersection* currentIntersection); int
    // computeIntersection(collision::geometry::Line&, collision::geometry::Sphere&, OutputVector*,
    // const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Line&, OutputVector*,
                            const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Point&,
                            OutputVector*,
                            const core::collision::Intersection* currentIntersection);
    // int computeIntersection(collision::geometry::Triangle&, collision::geometry::Sphere&,
    // OutputVector*, const core::collision::Intersection* currentIntersection); int
    // computeIntersection(collision::geometry::Ray&, collision::geometry::Sphere&, OutputVector*,
    // const core::collision::Intersection* currentIntersection); int
    // computeIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, OutputVector*,
    // const core::collision::Intersection* currentIntersection);


    Data<sofa::helper::OptionsGroup> d_continuousCollisionType;
    Data<SReal> d_tolerance;
    Data<long> d_maxIterations;
};
} // namespace sofa::component::collision::detection::intersection
