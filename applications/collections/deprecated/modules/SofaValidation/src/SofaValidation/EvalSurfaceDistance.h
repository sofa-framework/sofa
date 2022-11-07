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
#include <SofaValidation/config.h>

#include "EvalPointsDistance.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/component/collision/geometry/PointModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.h>
#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>

#include <fstream>

namespace sofa::component::misc
{


/** Compute the distance between surfaces in two objects
*/
template<class TDataTypes>
class EvalSurfaceDistance: public EvalPointsDistance<TDataTypes>
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(EvalSurfaceDistance,TDataTypes), SOFA_TEMPLATE(EvalPointsDistance,TDataTypes));

    typedef EvalPointsDistance<TDataTypes> Inherit;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    /// Alarm distance for proximity detection
    Data < SReal > maxDist;
protected:
    /** Default constructor
    */
    EvalSurfaceDistance();
    virtual ~EvalSurfaceDistance();
public:
    /// Compute the error metric between two surfaces
    SReal  eval() override;
    /// Init the computation
    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:

    /// Projection vector
    VecCoord xproj;

    /// Point model of first object
    sofa::component::collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types> *pointsCM;
    /// Surface model of second object
    sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types> *surfaceCM;

    sofa::component::collision::detection::intersection::NewProximityIntersection::SPtr intersection;
    sofa::component::collision::detection::algorithm::BVHNarrowPhase::SPtr narrowPhaseDetection;
    typedef core::collision::TDetectionOutputVector< sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, sofa::component::collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>> ContactVector;

};

#if  !defined(SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_CPP)
extern template class SOFA_SOFAVALIDATION_API EvalSurfaceDistance<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::misc
