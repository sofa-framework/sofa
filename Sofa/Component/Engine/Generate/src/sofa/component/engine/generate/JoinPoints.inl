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
#include <sofa/component/engine/generate/JoinPoints.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/system/FileRepository.h>
#include <list>

namespace sofa::component::engine::generate
{

template <class DataTypes>
JoinPoints<DataTypes>::JoinPoints()
    : f_points(initData(&f_points, "points", "Points"))
    , f_distance(initData(&f_distance, "distance", "Distance to merge points"))
    , f_mergedPoints(initData(&f_mergedPoints, "mergedPoints", "Merged Points"))
{
    addInput(&f_points);
    addInput(&f_distance);

    addOutput(&f_mergedPoints);
}

template <class DataTypes>
void JoinPoints<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void JoinPoints<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool JoinPoints<DataTypes>::getNearestPoint(const typename std::list<Coord>::iterator &itCurrentPoint,
        std::list<Coord>& listPoints,
        std::list<int>& listCoeffs,
        typename std::list<Coord>::iterator &itNearestPoint,
        std::list<int>::iterator &itNearestCoeff,
        const Real &distance)
{

    typename std::list<Coord>::iterator itPoint = listPoints.begin();

    std::list<int>::iterator itCoeff = listCoeffs.begin();

    Real min = 9999999.0;

    itNearestPoint = itCurrentPoint;

    for ( ; itPoint != listPoints.end() && itCoeff != listCoeffs.end(); itPoint++, itCoeff++)
    {
        if(itPoint != itCurrentPoint)
        {
            Real currentDist = (*itCurrentPoint - *itPoint).norm();
            if (currentDist < distance && min > currentDist)
            {
                min = currentDist;
                itNearestPoint = itPoint;
                itNearestCoeff = itCoeff;
            }
        }
    }

    return (itNearestPoint != itCurrentPoint);
}

template <class DataTypes>
void JoinPoints<DataTypes>::doUpdate()
{
    const VecCoord& points = f_points.getValue();
    const Real distance = f_distance.getValue();

    if (points.size() < 1)
    {
        msg_error() << "Error, no point defined";
        return ;
    }

    VecCoord& mergedPoints = *f_mergedPoints.beginEdit();
    mergedPoints.clear();

    std::list<Coord> copyPoints;
    std::list<int> coeffs;
    for (unsigned int i=0 ; i<points.size() ; i++)
    {
        copyPoints.push_back(points[i]);
        coeffs.push_back(1);
    }

    typename std::list<Coord>::iterator itCurrentPoint = copyPoints.begin();
    std::list<int>::iterator itCurrentCoeff = coeffs.begin();

    while (itCurrentPoint != copyPoints.end())
    {
        typename std::list<Coord>::iterator itNearestPoint;
        std::list<int>::iterator itNearestCoeff;
        const bool hasNearestPoint = getNearestPoint(itCurrentPoint, copyPoints, coeffs, itNearestPoint, itNearestCoeff, distance);

        //if we get a point in the sphere's ROI
        if (hasNearestPoint)
        {
            //compute new point
            (*itCurrentPoint) = (((*itCurrentPoint) * (*itCurrentCoeff)) + (*itNearestPoint)*(*itNearestCoeff))/((*itCurrentCoeff) + (*itNearestCoeff));
            (*itCurrentCoeff) = (*itCurrentCoeff) + (*itNearestCoeff);

            //delete the nearest Point in the list (and its coeff)
            copyPoints.erase(itNearestPoint);
            coeffs.erase(itNearestCoeff);
        }
        else
        {
            ++itCurrentPoint;
            ++itCurrentCoeff;
        }
    }

    //copy list into the result vector
    for (itCurrentPoint = copyPoints.begin(); itCurrentPoint != copyPoints.end() ; itCurrentPoint++)
        mergedPoints.push_back((*itCurrentPoint));

    f_mergedPoints.endEdit();
}

} //namespace sofa::component::engine::generate
