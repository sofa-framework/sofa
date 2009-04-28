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

namespace sofa
{

namespace component
{

namespace engine
{


template <class DataTypes>
Vertex2Frame<DataTypes>::Vertex2Frame():
    vertices(initData(&vertices,"vertices","Vertices of the mesh loaded"))
    , texCoords(initData(&texCoords,"texCoords","TexCoords of the mesh loaded"))
    , normals(initData(&normals,"normals","Normals of the mesh loaded"))
    , facets(initData(&facets,"facets","Facets of the mesh loaded"))
    , frames( initData (&frames, "frames", "Frames at output") )
{
    addInput(&vertices);
    addInput(&texCoords);
    addInput(&normals);
    addInput(&facets);

    addOutput(&frames);
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::update()
{
    dirty = false;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
