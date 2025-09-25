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
 * - Sphere/Sphere
 * - Sphere/Point
 * - Point/Point
 * - Line/Line
 * - Line/Point
 * - Line/Sphere
 * - Triangle/Point
 * - Triangle/Sphere
 * - Ray/Triangle
 * - Ray/Sphere
 * The following pairs of collision models are ignored:
 * - Triangle/Line
 * - Triangle/Triangle
 * - Ray/Point
 * - Ray/Line
 */
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API LocalMinDistance : public BaseProximityIntersection
{
public:
    SOFA_CLASS(LocalMinDistance,BaseProximityIntersection);

    typedef core::collision::IntersectorFactory<LocalMinDistance> IntersectorFactory;

    Data<bool> d_filterIntersection; ///< Activate LMD filter
    Data<double> d_angleCone; ///< Filtering cone extension angle
    Data<double> d_coneFactor; ///< Factor for filtering cone angle computation
    Data<bool> d_useLMDFilters; ///< Use external cone computation

protected:
    LocalMinDistance();

public:
    void init() override;
    
    bool testIntersection(collision::geometry::Cube& ,collision::geometry::Cube&, const core::collision::Intersection* currentIntersection) override;

    bool testIntersection(collision::geometry::Point&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Sphere&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Sphere&, collision::geometry::Sphere&, const core::collision::Intersection* currentIntersection) override;
    bool testIntersection(collision::geometry::Line&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Sphere&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Line&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Sphere&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Ray&, collision::geometry::Sphere&, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, const core::collision::Intersection* currentIntersection);

    int computeIntersection(collision::geometry::Cube&, collision::geometry::Cube&, OutputVector*, const core::collision::Intersection* currentIntersection) override;
    int computeIntersection(collision::geometry::Point&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Sphere&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Sphere&, collision::geometry::Sphere&, OutputVector*, const core::collision::Intersection* currentIntersection) override;
    int computeIntersection(collision::geometry::Line&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Sphere&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Line&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Sphere&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Ray&, collision::geometry::Sphere&, OutputVector*, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Ray&, collision::geometry::Triangle&, OutputVector*, const core::collision::Intersection* currentIntersection);

    /// These methods check the validity of a found intersection.
    /// According to the local configuration around the found intersected primitive,
    /// we build a "Region Of Interest" geometric cone.
    /// Pertinent intersections have to belong to this cone, others are not taking into account anymore.
    bool testValidity(collision::geometry::Sphere&, const type::Vec3&) const { return true; }
    bool testValidity(collision::geometry::Point&, const type::Vec3&) const;
    bool testValidity(collision::geometry::Line&, const type::Vec3&) const;
    bool testValidity(collision::geometry::Triangle&, const type::Vec3&) const;
};

} // namespace sofa::component::collision::detection::intersection
