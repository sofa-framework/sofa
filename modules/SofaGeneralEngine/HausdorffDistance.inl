/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_HAUSDORFFDISTANCE_INL
#define SOFA_COMPONENT_ENGINE_HAUSDORFFDISTANCE_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "HausdorffDistance.h"
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
HausdorffDistance<DataTypes>::HausdorffDistance()
    : f_points_1( initData (&f_points_1, "points1", "Points belonging to the first point cloud") )
    , f_points_2( initData (&f_points_2, "points2", "Points belonging to the second point cloud") )
    , d12( initData (&d12, "d12", "Distance from point cloud 1 to 2") )
    , d21( initData (&d21, "d21", "Distance from point cloud 2 to 1") )
    , max( initData (&max, "max", "Symmetrical Hausdorff distance") )
    , f_update( initData (&f_update, false, "update", "Recompute every time step") )
{
    f_points_1.setGroup("Input");
    f_points_2.setGroup("Input");

    d21.setGroup("Output");
    d12.setGroup("Output");
    max.setGroup("Output");

    f_listening.setValue(true);

}

template <class DataTypes>
void HausdorffDistance<DataTypes>::init()
{
    computeDistances();
}

template <class DataTypes>
void HausdorffDistance<DataTypes>::reinit()
{

}

template <class DataTypes>
void HausdorffDistance<DataTypes>::update()
{
    cleanDirty();

    if (f_update.getValue())
        computeDistances();
}

/**
 * Compute the distance from a point to a point cloud
 */
template <class DataTypes>
typename HausdorffDistance<DataTypes>::Real HausdorffDistance<DataTypes>::distance(Coord p, VecCoord S)
{
    Real min = std::numeric_limits<Real>::max();

    for (unsigned int i = 0 ; i < S.size(); i++)
    {
        Real d = (p-S[i]).norm();
        if (d<min) min = d;
    }

    return min;
}

/**
 * Compute distances between both point clouds (symmetrical and non-symmetrical distances)
 */
template <class DataTypes>
void HausdorffDistance<DataTypes>::computeDistances()
{
    const VecCoord p1 = f_points_1.getValue();
    const VecCoord p2 = f_points_2.getValue();

    Real max12 = 0.0;
    for (unsigned int i = 0 ; i < p1.size(); i++)
    {
        Real d = distance(p1[i], p2);
        if (d>max12) max12 = d;
    }
    d12.setValue(max12);

    Real max21 = 0.0;
    for (unsigned int i = 0 ; i < p2.size(); i++)
    {
        Real d = distance(p2[i], p1);
        if (d>max21) max21 = d;
    }
    d21.setValue(max21);

    if (max21 > max12)
        max.setValue(max21);
    else
        max.setValue(max12);
}

template<class DataTypes>
void HausdorffDistance<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
        this->onBeginAnimationStep(this->getContext()->getDt());
}

template <class DataTypes>
void HausdorffDistance<DataTypes>::onBeginAnimationStep(const double /*dt*/)
{
        update();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_ENGINE_HAUSDORFFDISTANCE_INL
