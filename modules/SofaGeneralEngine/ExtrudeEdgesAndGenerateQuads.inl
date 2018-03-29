/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_INL
#define SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/ExtrudeEdgesAndGenerateQuads.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using sofa::core::topology::BaseMeshTopology;

template <class DataTypes>
ExtrudeEdgesAndGenerateQuads<DataTypes>::ExtrudeEdgesAndGenerateQuads()
    : initialized(false)
    , d_direction( initData (&d_direction, Coord(1.0f,0.0f,0.0f), "extrudeDirection", "Direction along which to extrude the curve") )
    , d_thicknessIn( initData (&d_thicknessIn, Real (0.0), "thicknessIn", "Thickness of the extruded volume in the opposite direction of the normals") )
    , d_thicknessOut( initData (&d_thicknessOut, Real (1.0), "thicknessOut", "Thickness of the extruded volume in the direction of the normals") )
    , d_nbSections( initData (&d_nbSections, int (1), "numberOfSections", "Number of sections / steps in the extrusion") )
    , d_curveVertices( initData (&d_curveVertices, "curveVertices", "Position coordinates along the initial curve") )
    , d_curveEdges( initData (&d_curveEdges, "curveEdges", "Indices of the edges of the curve to extrude") )
    , d_extrudedVertices( initData (&d_extrudedVertices, "extrudedVertices", "Coordinates of the extruded vertices") )
    , d_extrudedEdges( initData (&d_extrudedEdges, "extrudedEdges", "List of all edges generated during the extrusion") )
    , d_extrudedQuads( initData (&d_extrudedQuads, "extrudedQuads", "List of all quads generated during the extrusion") )
{
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::init()
{
    addInput(&d_curveVertices);
    addInput(&d_curveEdges);

    addOutput(&d_extrudedVertices);
    addOutput(&d_extrudedEdges);
    addOutput(&d_extrudedQuads);

    setDirtyValue();
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::reinit()
{
    checkInput();
    update();
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::bwdInit()
{
    checkInput();
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::checkInput()
{
    if (d_curveEdges.getValue().size() < 1 || d_curveVertices.getValue().size() < 1)
        msg_warning(this) << "Initial mesh does not contain vertices or edges... No extruded mesh will be generated";

    if (d_nbSections.getValue() < 0 )
    {
        msg_warning(this) << "The number of sections should be positive. Set default equal to 1.";
        d_nbSections.setValue(1);
    }
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::update()
{
    const vector<BaseMeshTopology::Edge>& curveEdges = d_curveEdges.getValue();
    const VecCoord& curveVertices = d_curveVertices.getValue();

    if (curveEdges.size() < 1 || curveVertices.size() < 1)
    {

        msg_warning() << "Initial mesh does not contain vertices or edges... No extruded mesh will be generated";

        return;
    }

    cleanDirty();

    VecCoord* extrudedVertices = d_extrudedVertices.beginWriteOnly();
    extrudedVertices->clear();
    vector<BaseMeshTopology::Edge>* extrudedEdges = d_extrudedEdges.beginWriteOnly();
    extrudedEdges->clear();
    vector<BaseMeshTopology::Quad>* extrudedQuads = d_extrudedQuads.beginWriteOnly();
    extrudedQuads->clear();

    int nbSections = d_nbSections.getValue();
    int nbVertices = curveVertices.size();

    // compute coordinates of extruded vertices (including initial vertices)
    Real scale = (d_thicknessIn.getValue() + d_thicknessOut.getValue())/(Real)nbSections;
    for (int n=0; n<nbSections+1; n++)
    {
        Coord step = d_direction.getValue();
        step.normalize();
        step = step * scale;
        for (unsigned int i=0; i<curveVertices.size(); i++)
        {
            Coord disp = -d_direction.getValue() * d_thicknessIn.getValue() + step * (Real)n;
            Coord newVertexPos(curveVertices[i][0], curveVertices[i][1], curveVertices[i][2]);
            extrudedVertices->push_back(newVertexPos + disp);
        }
    }

    // compute indices of extruded edges (including initial edges)
    for (unsigned int e=0; e<curveEdges.size(); e++)
    {
        BaseMeshTopology::Edge edge;

        // edges parallel to the initial curve
        for (int n=0; n<nbSections+1; n++)
        {
            edge = BaseMeshTopology::Edge(curveEdges[e][0]+n*nbVertices, curveEdges[e][1]+n*nbVertices);
            extrudedEdges->push_back(edge);
        }
        // edges orthogonal to the initial curve
        for (int n=0; n<nbSections; n++)
        {
            edge = BaseMeshTopology::Edge(curveEdges[e][0]+n*nbVertices, curveEdges[e][0]+(n+1)*nbVertices);
            extrudedEdges->push_back(edge);
        }
    }

    // compute indices of extruded quads
    for (unsigned int e=0; e<curveEdges.size(); e++)
    {
        BaseMeshTopology::Quad quad;

        for (int n=0; n<nbSections; n++)
        {
            quad = BaseMeshTopology::Quad(curveEdges[e][0]+n*nbVertices, curveEdges[e][1]+n*nbVertices, curveEdges[e][1]+(n+1)*nbVertices, curveEdges[e][0]+(n+1)*nbVertices);
            extrudedQuads->push_back(quad);
        }
    }

    // curve not closed
    if(curveEdges.size() < curveVertices.size())
    {
        int e = curveEdges.size()-1;
        for (int n=0; n<nbSections; n++)
        {
            BaseMeshTopology::Edge edge = BaseMeshTopology::Edge(curveEdges[e][1]+n*nbVertices, curveEdges[e][1]+(n+1)*nbVertices);
            extrudedEdges->push_back(edge);
        }
    }

    d_extrudedQuads.endEdit();
    d_extrudedEdges.endEdit();
    d_extrudedVertices.endEdit();
}




} // namespace engine

} // namespace component

} // namespace sofa

#endif
