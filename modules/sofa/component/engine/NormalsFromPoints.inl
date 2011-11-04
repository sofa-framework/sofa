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
#ifndef SOFA_COMPONENT_ENGINE_NormalsFromPoints_INL
#define SOFA_COMPONENT_ENGINE_NormalsFromPoints_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "NormalsFromPoints.h"
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;

template <class DataTypes>
NormalsFromPoints<DataTypes>::NormalsFromPoints()
    :   position(initData(&position,"position","Vertices of the mesh"))
    , triangles(initData(&triangles,"triangles","Triangles of the mesh"))
    , quads(initData(&quads,"quads","Quads of the mesh"))
    , normals(initData(&normals,"normals","Computed vertex normals of the mesh"))
    , invertNormals( initData (&invertNormals, false, "invertNormals", "Swap normals") )
{
}

template <class DataTypes>
void NormalsFromPoints<DataTypes>::init()
{
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&position);
    addInput(&triangles);
    addInput(&quads);
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
    cleanDirty();

    helper::ReadAccessor<Data< VecCoord > > raPositions = position;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,3> > > > raTriangles = triangles;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > raQuads = quads;
    helper::WriteAccessor<Data< VecCoord > > waNormals = normals;

    waNormals.resize(raPositions.size());

    for (unsigned int i = 0; i < raTriangles.size() ; i++)
    {
        const Coord  v1 = raPositions[raTriangles[i][0]];
        const Coord  v2 = raPositions[raTriangles[i][1]];
        const Coord  v3 = raPositions[raTriangles[i][2]];
        Coord n = cross(v2-v1, v3-v1);

        n.normalize();
        waNormals[raTriangles[i][0]] += n;
        waNormals[raTriangles[i][1]] += n;
        waNormals[raTriangles[i][2]] += n;

    }
    for (unsigned int i = 0; i < raQuads.size() ; i++)
    {
        const Coord & v1 = raPositions[raQuads[i][0]];
        const Coord & v2 = raPositions[raQuads[i][1]];
        const Coord & v3 = raPositions[raQuads[i][2]];
        const Coord & v4 = raPositions[raQuads[i][3]];
        Coord n1 = cross(v2-v1, v4-v1);
        Coord n2 = cross(v3-v2, v1-v2);
        Coord n3 = cross(v4-v3, v2-v3);
        Coord n4 = cross(v1-v4, v3-v4);
        n1.normalize(); n2.normalize(); n3.normalize(); n4.normalize();
        waNormals[raQuads[i][0]] += n1;
        waNormals[raQuads[i][1]] += n2;
        waNormals[raQuads[i][2]] += n3;
        waNormals[raQuads[i][3]] += n4;
    }

    if(invertNormals.getValue())	  for (unsigned int i = 0; i < waNormals.size(); i++)		  waNormals[i]*=-1.;
    for (unsigned int i = 0; i < waNormals.size(); i++)		  waNormals[i].normalize();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
