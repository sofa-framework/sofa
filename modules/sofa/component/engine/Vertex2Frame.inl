/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_VERTEX2FRAME_INL
#define SOFA_COMPONENT_ENGINE_VERTEX2FRAME_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/Vertex2Frame.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

template <class DataTypes>
Vertex2Frame<DataTypes>::Vertex2Frame():
    vertices(initData(&vertices,"position","Vertices of the mesh loaded"))
    , texCoords(initData(&texCoords,"texCoords","TexCoords of the mesh loaded"))
    , normals(initData(&normals,"normals","Normals of the mesh loaded"))
    , frames( initData (&frames, "frames", "Frames at output") )
    , invertNormals( initData (&invertNormals, false, "invertNormals", "Swap normals") )
{
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::init()
{
    addInput(&vertices);
    addInput(&texCoords);
    addInput(&normals);

    addOutput(&frames);

    setDirtyValue();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::update()
{
    cleanDirty();

    const helper::vector<Vector3>& fVertices = vertices.getValue();
    const helper::vector<Vector3>& fNormals = normals.getValue();
    unsigned int nbVertices = fVertices.size();

    if (nbVertices <= 0 || fNormals.size() <=0)
    {
        serr << "Vertex2Frame : no vertices or normals found..." << sendl;
        return ;
    }

    VecCoord& fFrames = *(frames.beginEdit());
    fFrames.resize(nbVertices);

    for (unsigned int i=0 ; i<nbVertices ; i++)
    {
        Quat q;
        Vector3 zAxis = (!invertNormals.getValue()) ? fNormals[i] : -fNormals[i];
        zAxis.normalize();
        Vector3 xAxis;
        Vector3 yAxis(1.0, 0.0, 0.0);
        if ( fabs(dot(yAxis, zAxis)) > 0.7)
            yAxis = Vector3(0.0, 0.0, 1.0);

        xAxis = yAxis.cross(zAxis);
        xAxis.normalize();
        yAxis = zAxis.cross(xAxis);
        yAxis.normalize();

        fFrames[i].getOrientation() = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
        fFrames[i].getCenter() = fVertices[i];
    }

    frames.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
