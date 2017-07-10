/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_ENGINE_NormalsFromPoints_INL
#define SOFA_COMPONENT_ENGINE_NormalsFromPoints_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "NormalsFromPoints.h"
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <math.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
NormalsFromPoints<DataTypes>::NormalsFromPoints()
    : position(initData(&position,"position","Vertices of the mesh"))
    , triangles(initData(&triangles,"triangles","Triangles of the mesh"))
    , quads(initData(&quads,"quads","Quads of the mesh"))
    , normals(initData(&normals,"normals","Computed vertex normals of the mesh"))
    , invertNormals( initData (&invertNormals, false, "invertNormals", "Swap normals") )
    , useAngles( initData (&useAngles, false, "useAngles", "Use incident angles to weight faces normal contributions at each vertex") )

{
}

template <class DataTypes>
void NormalsFromPoints<DataTypes>::init()
{
    addInput(&position);
    addInput(&triangles);
    addInput(&quads);
    addInput(&invertNormals);
    addInput(&useAngles);
    addOutput(&normals);
    setDirtyValue();
}

template <class DataTypes>
void NormalsFromPoints<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void NormalsFromPoints<DataTypes>::update()
{
    helper::ReadAccessor<Data< VecCoord > > raPositions = position;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,3> > > > raTriangles = triangles;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > raQuads = quads;
    helper::WriteOnlyAccessor<Data< VecCoord > > waNormals = normals;
    const bool useAngles = this->useAngles.getValue();
    const bool invertNormals = this->invertNormals.getValue();

    waNormals.resize(raPositions.size());

    for (unsigned int i = 0; i < raTriangles.size() ; i++)
    {
        const Coord  v1 = raPositions[raTriangles[i][0]];
        const Coord  v2 = raPositions[raTriangles[i][1]];
        const Coord  v3 = raPositions[raTriangles[i][2]];
        Coord n = cross(v2-v1, v3-v1).normalized();
        if (useAngles)
        {
            Coord e12 = (v2-v1).normalized();
            Coord e23 = (v3-v2).normalized();
            Coord e31 = (v1-v3).normalized();
            waNormals[raTriangles[i][0]] += n * acos(-(e31*e12));
            waNormals[raTriangles[i][1]] += n * acos(-(e12*e23));
            waNormals[raTriangles[i][2]] += n * acos(-(e23*e31));
        }
        else
        {
            waNormals[raTriangles[i][0]] += n;
            waNormals[raTriangles[i][1]] += n;
            waNormals[raTriangles[i][2]] += n;
        }
    }
    for (unsigned int i = 0; i < raQuads.size() ; i++)
    {
        const Coord & v1 = raPositions[raQuads[i][0]];
        const Coord & v2 = raPositions[raQuads[i][1]];
        const Coord & v3 = raPositions[raQuads[i][2]];
        const Coord & v4 = raPositions[raQuads[i][3]];
        Coord n1 = cross(v2-v1, v4-v1).normalized();
        Coord n2 = cross(v3-v2, v1-v2).normalized();
        Coord n3 = cross(v4-v3, v2-v3).normalized();
        Coord n4 = cross(v1-v4, v3-v4).normalized();
        if (useAngles)
        {
            Coord e12 = (v2-v1).normalized();
            Coord e23 = (v3-v2).normalized();
            Coord e34 = (v4-v3).normalized();
            Coord e41 = (v1-v4).normalized();
            waNormals[raQuads[i][0]] += n1 * acos(-(e41*e12));
            waNormals[raQuads[i][1]] += n2 * acos(-(e12*e23));
            waNormals[raQuads[i][2]] += n3 * acos(-(e23*e34));
            waNormals[raQuads[i][3]] += n4 * acos(-(e34*e41));
        }
        else
        {
            waNormals[raQuads[i][0]] += n1;
            waNormals[raQuads[i][1]] += n2;
            waNormals[raQuads[i][2]] += n3;
            waNormals[raQuads[i][3]] += n4;
        }
    }

    if(invertNormals)
        for (unsigned int i = 0; i < waNormals.size(); i++)
            waNormals[i]=-waNormals[i];

    const Coord failsafe = Coord(1,1,1).normalized();
    for (unsigned int i = 0; i < waNormals.size(); i++)
        waNormals[i].normalize(failsafe);

    cleanDirty();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
