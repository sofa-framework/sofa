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

#include <sofa/component/collision/model/SphereModel.h>
#include <sofa/component/collision/model/TriangleModel.h>
#include <sofa/component/collision/model/LineModel.h>
#include <sofa/component/collision/model/PointModel.h>
#include <sofa/component/collision/model/CubeModel.h>
#include <sofa/component/collision/model/RayModel.h>

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

    Data<bool> filterIntersection; ///< Activate LMD filter
    Data<double> angleCone; ///< Filtering cone extension angle
    Data<double> coneFactor; ///< Factor for filtering cone angle computation
    Data<bool> useLMDFilters; ///< Use external cone computation (Work in Progress)

protected:
    LocalMinDistance();

public:
    void init() override;

    bool testIntersection(model::Cube& ,model::Cube&) override;

    bool testIntersection(model::Point&, model::Point&);
    bool testIntersection(model::Sphere&, model::Point&);
    bool testIntersection(model::Sphere&, model::Sphere&) override;
    bool testIntersection(model::Line&, model::Point&);
    bool testIntersection(model::Line&, model::Sphere&);
    bool testIntersection(model::Line&, model::Line&);
    bool testIntersection(model::Triangle&, model::Point&);
    bool testIntersection(model::Triangle&, model::Sphere&);
    bool testIntersection(model::Ray&, model::Sphere&);
    bool testIntersection(model::Ray&, model::Triangle&);

    int computeIntersection(model::Cube&, model::Cube&, OutputVector*) override;
    int computeIntersection(model::Point&, model::Point&, OutputVector*);
    int computeIntersection(model::Sphere&, model::Point&, OutputVector*);
    int computeIntersection(model::Sphere&, model::Sphere&, OutputVector*) override;
    int computeIntersection(model::Line&, model::Point&, OutputVector*);
    int computeIntersection(model::Line&, model::Sphere&, OutputVector*);
    int computeIntersection(model::Line&, model::Line&, OutputVector*);
    int computeIntersection(model::Triangle&, model::Point&, OutputVector*);
    int computeIntersection(model::Triangle&, model::Sphere&, OutputVector*);
    int computeIntersection(model::Ray&, model::Sphere&, OutputVector*);
    int computeIntersection(model::Ray&, model::Triangle&, OutputVector*);

    /// These methods check the validity of a found intersection.
    /// According to the local configuration around the found intersected primitive,
    /// we build a "Region Of Interest" geometric cone.
    /// Pertinent intersections have to belong to this cone, others are not taking into account anymore.
    bool testValidity(model::Sphere&, const type::Vector3&) const { return true; }
    bool testValidity(model::Point&, const type::Vector3&) const;
    bool testValidity(model::Line&, const type::Vector3&) const;
    bool testValidity(model::Triangle&, const type::Vector3&) const;

};

} // namespace sofa::component::collision::detection::intersection
