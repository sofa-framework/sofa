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
#ifndef SOFA_COMPONENT_ENGINE_MERGEPOINTS_INL
#define SOFA_COMPONENT_ENGINE_MERGEPOINTS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/MergePoints.h>
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
    , f_X2_mapping( initData (&f_X2_mapping, "mappingX2", "Mapping of indices to inject position2 inside position1 vertex buffer") )
    , f_indices1( initData(&f_indices1,"indices1","Indices of the points of the first object") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the points of the second object") )
    , f_points( initData (&f_points, "points", "position coordinates of the merge") )
    , f_noUpdate( initData (&f_noUpdate, false, "noUpdate", "do not update the output at eacth time step (false)") )
{
}

template <class DataTypes>
void MergePoints<DataTypes>::init()
{
    addInput(&f_X1);
    addInput(&f_X2);
    addInput(&f_X2_mapping);
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
    if (f_noUpdate.getValue() && initDone)
        return;

    const VecCoord& x1 = f_X1.getValue();
    const VecCoord& x2 = f_X2.getValue();


    // get access to output buffers
    SetIndex& indices1 = *(f_indices1.beginWriteOnly());
    SetIndex& indices2 = *(f_indices2.beginWriteOnly());
    VecCoord& points = *(f_points.beginWriteOnly());

    // clear buffers
    indices1.clear();
    indices2.clear();
    points.clear();

    if (f_X2_mapping.isSet() && !f_X2_mapping.getValue().empty()) // mode injection
    {                
        // mapping of X2
        sofa::helper::vector <unsigned int> mapping = f_X2_mapping.getValue();

        // fill buffer1 to full X1
        points = x1;

        for(unsigned int i=0; i<mapping.size(); ++i)
        {
            unsigned int posX = mapping[i];
            if (posX < points.size()) // new point to insert
                points[posX] = x2[i]; // insert X2 inside X1
            else
                serr << "Error Trying to insert vertex from mapping at pos: " <<  posX << " which is out of bounds of X1." << sendl;
        }

        // fill indice1 & indice2 buffers
        std::sort(mapping.begin(), mapping.end());
        int j=0;
        for(unsigned int i=0; i<points.size(); ++i)
        {
            unsigned int posX = mapping[j];
            if(i == posX)
            {
                indices2.push_back(i); // fill indices2 buffer
                j++;
            }
            else // fill indice1 buffer
                indices1.push_back(i);
        }
    }
    else // mode addition
    {
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
    }

    cleanDirty();

    f_indices1.endEdit();
    f_indices2.endEdit();
    f_points.endEdit();

    initDone=true;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
