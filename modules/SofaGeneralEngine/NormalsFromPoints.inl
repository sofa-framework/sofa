/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
        Coord n = cross(v2-v1, v3-v1);
        if (useAngles)
        {
            Real nnorm = n.norm();
            Coord e12 = v2-v1; Real e12norm = e12.norm();
            Coord e23 = v3-v2; Real e23norm = e23.norm();
            Coord e31 = v1-v3; Real e31norm = e31.norm();
            waNormals[raTriangles[i][0]] += n * (acos(-(e31*e12)/(e31norm*e12norm))/nnorm);
            waNormals[raTriangles[i][1]] += n * (acos(-(e12*e23)/(e12norm*e23norm))/nnorm);
            waNormals[raTriangles[i][2]] += n * (acos(-(e23*e31)/(e23norm*e31norm))/nnorm);
        }
        else
        {
            n.normalize();
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
        Coord n1 = cross(v2-v1, v4-v1); Real n1norm = n1.norm();
        Coord n2 = cross(v3-v2, v1-v2); Real n2norm = n2.norm();
        Coord n3 = cross(v4-v3, v2-v3); Real n3norm = n3.norm();
        Coord n4 = cross(v1-v4, v3-v4); Real n4norm = n4.norm();
        if (useAngles)
        {
            Coord e12 = v2-v1; Real e12norm = e12.norm();
            Coord e23 = v3-v2; Real e23norm = e23.norm();
            Coord e34 = v4-v3; Real e34norm = e34.norm();
            Coord e41 = v1-v4; Real e41norm = e41.norm();
            waNormals[raQuads[i][0]] += n1 * (acos(-(e41*e12)/(e41norm*e12norm))/n1norm);
            waNormals[raQuads[i][1]] += n2 * (acos(-(e12*e23)/(e12norm*e23norm))/n2norm);
            waNormals[raQuads[i][2]] += n3 * (acos(-(e23*e34)/(e23norm*e34norm))/n3norm);
            waNormals[raQuads[i][3]] += n4 * (acos(-(e34*e41)/(e34norm*e41norm))/n3norm);
        }
        else
        {
            waNormals[raQuads[i][0]] += n1 / n1norm;
            waNormals[raQuads[i][1]] += n2 / n2norm;
            waNormals[raQuads[i][2]] += n3 / n3norm;
            waNormals[raQuads[i][3]] += n4 / n4norm;
        }
    }

    if(invertNormals)
        for (unsigned int i = 0; i < waNormals.size(); i++)
            waNormals[i]=-waNormals[i];

    for (unsigned int i = 0; i < waNormals.size(); i++)
        waNormals[i].normalize();

    cleanDirty();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
