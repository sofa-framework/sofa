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
#include <sofa/component/engine/analyze/HausdorffDistance.h>
#include <iostream>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa::component::engine::analyze
{

template <class DataTypes>
HausdorffDistance<DataTypes>::HausdorffDistance()
    : d_points_1(initData (&d_points_1, "points1", "Points belonging to the first point cloud") )
    , d_points_2(initData (&d_points_2, "points2", "Points belonging to the second point cloud") )
    , d_d12(initData (&d_d12, "d12", "Distance from point cloud 1 to 2") )
    , d_d21(initData (&d_d21, "d21", "Distance from point cloud 2 to 1") )
    , d_max(initData (&d_max, "max", "Symmetrical Hausdorff distance") )
    , d_update(initData (&d_update, false, "update", "Recompute every time step") )
{
    d_points_1.setGroup("Input");
    d_points_2.setGroup("Input");

    d_d21.setGroup("Output");
    d_d12.setGroup("Output");
    d_max.setGroup("Output");

    f_listening.setValue(true);

    f_points_1.setOriginalData(&d_points_1);
    f_points_2.setOriginalData(&d_points_2);
    d12.setOriginalData(&d_d12);
    d21.setOriginalData(&d_d21);
    max.setOriginalData(&d_max);
    f_update.setOriginalData(&d_update);


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
void HausdorffDistance<DataTypes>::doUpdate()
{
    if (d_update.getValue())
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
    const VecCoord p1 = d_points_1.getValue();
    const VecCoord p2 = d_points_2.getValue();

    Real max12 = 0.0;
    for (unsigned int i = 0 ; i < p1.size(); i++)
    {
        Real d = distance(p1[i], p2);
        if (d>max12) max12 = d;
    }
    d_d12.setValue(max12);

    Real max21 = 0.0;
    for (unsigned int i = 0 ; i < p2.size(); i++)
    {
        Real d = distance(p2[i], p1);
        if (d>max21) max21 = d;
    }
    d_d21.setValue(max21);

    if (max21 > max12)
        d_max.setValue(max21);
    else
        d_max.setValue(max12);
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

} //namespace sofa::component::engine::analyze
