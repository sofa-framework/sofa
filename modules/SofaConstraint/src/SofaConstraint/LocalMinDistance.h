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
#include <SofaConstraint/config.h>

#include <SofaBaseCollision/BaseProximityIntersection.h>

#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>

namespace sofa::component::collision
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
class SOFA_SOFACONSTRAINT_API LocalMinDistance : public BaseProximityIntersection
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

    bool testIntersection(Cube& ,Cube&) override;

    bool testIntersection(Point&, Point&);
    bool testIntersection(Sphere&, Point&);
    bool testIntersection(Sphere&, Sphere&) override;
    bool testIntersection(Line&, Point&);
    bool testIntersection(Line&, Sphere&);
    bool testIntersection(Line&, Line&);
    bool testIntersection(Triangle&, Point&);
    bool testIntersection(Triangle&, Sphere&);
    bool testIntersection(Ray&, Sphere&);
    bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, OutputVector*) override;
    int computeIntersection(Point&, Point&, OutputVector*);
    int computeIntersection(Sphere&, Point&, OutputVector*);
    int computeIntersection(Sphere&, Sphere&, OutputVector*) override;
    int computeIntersection(Line&, Point&, OutputVector*);
    int computeIntersection(Line&, Sphere&, OutputVector*);
    int computeIntersection(Line&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Point&, OutputVector*);
    int computeIntersection(Triangle&, Sphere&, OutputVector*);
    int computeIntersection(Ray&, Sphere&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);

    /// These methods check the validity of a found intersection.
    /// According to the local configuration around the found intersected primitive,
    /// we build a "Region Of Interest" geometric cone.
    /// Pertinent intersections have to belong to this cone, others are not taking into account anymore.
    bool testValidity(Sphere&, const type::Vector3&) const { return true; }
    bool testValidity(Point&, const type::Vector3&) const;
    bool testValidity(Line&, const type::Vector3&) const;
    bool testValidity(Triangle&, const type::Vector3&) const;

};

} // namespace sofa::component::collision
