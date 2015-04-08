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
#ifndef SOFA_COMPONENT_ENGINE_MERGEPOINTS_INL
#define SOFA_COMPONENT_ENGINE_MERGEPOINTS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaEngine/MergePoints.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
MergePoints<DataTypes>::MergePoints()
    : f_X1( initData (&f_X1, "position1", "position coordinates of the degrees of freedom of the first object") )
    , f_X2( initData (&f_X2, "position2", "Rest position coordinates of the degrees of freedom of the second object") )
    , f_indices1( initData(&f_indices1,"indices1","Indices of the points of the first object") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the points of the second object") )
    , f_points( initData (&f_points, "points", "position coordinates of the merge") )
{
}

template <class DataTypes>
void MergePoints<DataTypes>::init()
{
    addInput(&f_X1);
    addInput(&f_X2);
    addOutput(&f_indices1);
    addOutput(&f_indices2);
    addOutput(&f_points);
    setDirtyValue();
}

template <class DataTypes>
void MergePoints<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void MergePoints<DataTypes>::update()
{
    cleanDirty();

    const VecCoord& x1 = f_X1.getValue();
    const VecCoord& x2 = f_X2.getValue();

    SetIndex& indices1 = *(f_indices1.beginEdit());
    SetIndex& indices2 = *(f_indices2.beginEdit());

    VecCoord& points = *(f_points.beginEdit());

    indices1.clear();
    indices2.clear();
    points.clear();

    for( unsigned i=0; i<x1.size(); ++i )
    {
        points.push_back(x1[i]);
        indices1.push_back(i);
    }

    unsigned int index = indices1.size();

    for( unsigned i=0; i<x2.size(); ++i )
    {
        points.push_back(x2[i]);
        indices2.push_back(index+i);
    }

    f_indices1.endEdit();
    f_indices2.endEdit();
    f_points.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
