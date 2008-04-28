/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_H
#define SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_H

#include "EvalPointsDistance.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/tree/AnimateBeginEvent.h>
#include <sofa/simulation/tree/AnimateEndEvent.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/BruteForceDetection.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class TDataTypes>
class EvalSurfaceDistance: public EvalPointsDistance<TDataTypes>, public virtual sofa::core::objectmodel::BaseObject
{
public:
    typedef EvalPointsDistance<TDataTypes> Inherit;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    Data < Real_Sofa > maxDist;
    EvalSurfaceDistance();
    virtual ~EvalSurfaceDistance();

    virtual Real_Sofa  eval();
    virtual void init();
    virtual void draw();

protected:
    VecCoord xproj;

    sofa::component::collision::TriangleModel *surfaceCM;
    sofa::component::collision::PointModel *pointsCM;
    //sofa::component::collision::MinProximityIntersection * intersection;
    sofa::component::collision::NewProximityIntersection * intersection;
    sofa::component::collision::BruteForceDetection* detection;
    typedef core::componentmodel::collision::TDetectionOutputVector< sofa::component::collision::TriangleModel, sofa::component::collision::PointModel> ContactVector;

};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
