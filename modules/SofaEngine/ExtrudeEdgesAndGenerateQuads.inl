/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_INL
#define SOFA_COMPONENT_ENGINE_EXTRUDEEDGESANDGENERATEQUADS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaEngine/ExtrudeEdgesAndGenerateQuads.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
ExtrudeEdgesAndGenerateQuads<DataTypes>::ExtrudeEdgesAndGenerateQuads()
    : initialized(false)
    , isVisible( initData (&isVisible, bool (true), "isVisible", "is Visible ?") )
    , f_direction( initData (&f_direction, Coord(1.0f,0.0f,0.0f), "extrudeDirection", "Direction along which to extrude the curve") )
    , f_thicknessIn( initData (&f_thicknessIn, Real (0.0), "thicknessIn", "Thickness of the extruded volume in the opposite direction of the normals") )
    , f_thicknessOut( initData (&f_thicknessOut, Real (1.0), "thicknessOut", "Thickness of the extruded volume in the direction of the normals") )
    , f_numberOfSections( initData (&f_numberOfSections, int (1), "numberOfSections", "Number of sections / steps in the extrusion") )
    , f_curveVertices( initData (&f_curveVertices, "curveVertices", "Position coordinates along the initial curve") )
    , f_curveEdges( initData (&f_curveEdges, "curveEdges", "Indices of the edges of the curve to extrude") )
    , f_extrudedVertices( initData (&f_extrudedVertices, "extrudedVertices", "Coordinates of the extruded vertices") )
    , f_extrudedQuads( initData (&f_extrudedQuads, "extrudedQuads", "List of all quads generated during the extrusion") )
{
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::init()
{
    addInput(&f_curveVertices);
    addInput(&f_curveEdges);

    addOutput(&f_extrudedVertices);
    addOutput(&f_extrudedQuads);

    setDirtyValue();
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ExtrudeEdgesAndGenerateQuads<DataTypes>::update()
{
    const helper::vector<sofa::core::topology::BaseMeshTopology::Edge>& curveEdges = f_curveEdges.getValue();
    const VecCoord& curveVertices = f_curveVertices.getValue();

    if (curveEdges.size() < 1 || curveVertices.size() < 1)
    {
        std::cout << "WARNING: initial mesh does not contain vertices or edges... No extruded mesh will be generated" << std::endl;
        return;
    }

    cleanDirty();

    VecCoord* extrudedVertices = f_extrudedVertices.beginWriteOnly();
    extrudedVertices->clear();
    helper::vector<sofa::core::topology::BaseMeshTopology::Edge>* extrudedEdges = f_extrudedEdges.beginWriteOnly();
    extrudedEdges->clear();
    helper::vector<sofa::core::topology::BaseMeshTopology::Quad>* extrudedQuads = f_extrudedQuads.beginWriteOnly();
    extrudedQuads->clear();

    int nSlices = f_numberOfSections.getValue();
    int nVertices = curveVertices.size();

    // compute coordinates of extruded vertices (including initial vertices)
    Real scale = (f_thicknessIn.getValue() + f_thicknessOut.getValue())/(Real)f_numberOfSections.getValue();
    for (int n=0; n<nSlices+1; n++)
    {
        Coord step = f_direction.getValue();
        step.normalize();
        step = step * scale;
        for (unsigned int i=0; i<curveVertices.size(); i++)
        {
            Coord disp = -f_direction.getValue() * f_thicknessIn.getValue() + step * (Real)n;
            Coord newVertexPos(curveVertices[i][0], curveVertices[i][1], curveVertices[i][2]);
            extrudedVertices->push_back(newVertexPos + disp);
        }
    }

    // compute indices of extruded edges (including initial edges)
    for (unsigned int e=0; e<curveEdges.size(); e++)
    {
        sofa::core::topology::BaseMeshTopology::Edge edge;

        // edges parallel to the initial curve
        for (int n=0; n<nSlices+1; n++)
        {
            edge = sofa::core::topology::BaseMeshTopology::Edge(curveEdges[e][0]+n*nVertices, curveEdges[e][1]+n*nVertices);
            extrudedEdges->push_back(edge);
        }
        // edges orthogonal to the initial curve
        for (int n=0; n<nSlices; n++)
        {
            edge = sofa::core::topology::BaseMeshTopology::Edge(curveEdges[e][0]+n*nVertices, curveEdges[e][0]+(n+1)*nVertices);
            extrudedEdges->push_back(edge);
        }
    }

    // compute indices of extruded quads
    for (unsigned int e=0; e<curveEdges.size(); e++)
    {
        sofa::core::topology::BaseMeshTopology::Quad quad;

        for (int n=0; n<nSlices; n++)
        {
            quad = sofa::core::topology::BaseMeshTopology::Quad(curveEdges[e][0]+n*nVertices, curveEdges[e][1]+n*nVertices, curveEdges[e][1]+(n+1)*nVertices, curveEdges[e][0]+(n+1)*nVertices);
            extrudedQuads->push_back(quad);
        }
    }

    // std::cout << "============= DONE ==========" << std::endl;

    f_extrudedQuads.endEdit();
    f_extrudedEdges.endEdit();
    f_extrudedVertices.endEdit();
}




} // namespace engine

} // namespace component

} // namespace sofa

#endif
