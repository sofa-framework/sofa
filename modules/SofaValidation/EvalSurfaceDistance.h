/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_H
#define SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_H

#include "EvalPointsDistance.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaBaseCollision/BruteForceDetection.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
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
    virtual SReal  eval();
    /// Init the computation
    virtual void init();
    virtual void draw(const core::visual::VisualParams* vparams);

protected:

    /// Projection vector
    VecCoord xproj;

    /// Point model of first object
    sofa::component::collision::PointModel *pointsCM;
    /// Surface model of second object
    sofa::component::collision::TriangleModel *surfaceCM;

    sofa::component::collision::NewProximityIntersection::SPtr intersection;
    sofa::component::collision::BruteForceDetection::SPtr detection;
    typedef core::collision::TDetectionOutputVector< sofa::component::collision::TriangleModel, sofa::component::collision::PointModel> ContactVector;

};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
