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
#ifndef SOFA_COMPONENT_ENGINE_EXTRUDEQUADSANDGENERATEHEXAS_INL
#define SOFA_COMPONENT_ENGINE_EXTRUDEQUADSANDGENERATEHEXAS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/ExtrudeQuadsAndGenerateHexas.h>
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
ExtrudeQuadsAndGenerateHexas<DataTypes>::ExtrudeQuadsAndGenerateHexas()
    : initialized(false)
    , isVisible( initData (&isVisible, bool (true), "isVisible", "is Visible ?") )
    , f_scale( initData (&f_scale, Coord(1.0f,1.0f,1.0f), "scale", "Apply a scaling factor to the extruded mesh") )
    , f_thicknessIn( initData (&f_thicknessIn, Real (0.0), "thicknessIn", "Thickness of the extruded volume in the opposite direction of the normals") )
    , f_thicknessOut( initData (&f_thicknessOut, Real (1.0), "thicknessOut", "Thickness of the extruded volume in the direction of the normals") )
    , f_numberOfSlices( initData (&f_numberOfSlices, int (1), "numberOfSlices", "Number of slices / steps in the extrusion") )
    , f_surfaceVertices( initData (&f_surfaceVertices, "surfaceVertices", "Position coordinates of the surface") )
    , f_surfaceQuads( initData (&f_surfaceQuads, "surfaceQuads", "Indices of the quads of the surface to extrude") )
    , f_extrudedVertices( initData (&f_extrudedVertices, "extrudedVertices", "Coordinates of the extruded vertices") )
    , f_extrudedSurfaceQuads( initData (&f_extrudedSurfaceQuads, "extrudedSurfaceQuads", "List of new surface quads generated during the extrusion") )
    , f_extrudedQuads( initData (&f_extrudedQuads, "extrudedQuads", "List of all quads generated during the extrusion") )
    , f_extrudedHexas( initData (&f_extrudedHexas, "extrudedHexas", "List of hexahedra generated during the extrusion") )
{
}

template <class DataTypes>
void ExtrudeQuadsAndGenerateHexas<DataTypes>::init()
{
    addInput(&f_surfaceQuads);
    addInput(&f_surfaceVertices);

    addOutput(&f_extrudedVertices);
    addOutput(&f_extrudedSurfaceQuads);
    addOutput(&f_extrudedQuads);
    addOutput(&f_extrudedHexas);

    setDirtyValue();
}

template <class DataTypes>
void ExtrudeQuadsAndGenerateHexas<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ExtrudeQuadsAndGenerateHexas<DataTypes>::update()
{
    using sofa::core::topology::BaseMeshTopology;

    const helper::vector<BaseMeshTopology::Quad>& surfaceQuads = f_surfaceQuads.getValue();
    const VecCoord& surfaceVertices = f_surfaceVertices.getValue();

    if (surfaceVertices.size() < 1 || surfaceQuads.size() < 1)
    {
        msg_warning() << "initial mesh does not contain vertices or quads... No extruded mesh will be generated" ;
        return;
    }

    cleanDirty();

    VecCoord* extrudedVertices = f_extrudedVertices.beginWriteOnly();
    extrudedVertices->clear();
    helper::vector<BaseMeshTopology::Quad>* extrudedSurfaceQuads = f_extrudedSurfaceQuads.beginWriteOnly();
    extrudedSurfaceQuads->clear();
    helper::vector<BaseMeshTopology::Quad>* extrudedQuads = f_extrudedQuads.beginWriteOnly();
    extrudedQuads->clear();
    helper::vector<BaseMeshTopology::Hexa>* extrudedHexas = f_extrudedHexas.beginWriteOnly();
    extrudedHexas->clear();

    std::map<int, std::pair<Vec3, unsigned int> > normals;
    int nSlices = f_numberOfSlices.getValue();

    //first loop to compute normals per point
    for (unsigned int q=0; q<surfaceQuads.size(); q++)
    {
        VecCoord quadCoord;

        //fetch real coords
        for (unsigned int i=0 ; i<4 ; i++)
            quadCoord.push_back(surfaceVertices[surfaceQuads[q][i]]);

        //compute normal
        Coord n =  cross(quadCoord[1]-quadCoord[0], quadCoord[2]-quadCoord[0]);
        n.normalize();
        for (int i=0 ; i<4 ; i++)
        {
            normals[surfaceQuads[q][i]].first += n;
            normals[surfaceQuads[q][i]].second++;
        }
    }

    //average normals
    typename std::map<int, std::pair<Vec3, unsigned int> >::iterator itNormals;
    for (itNormals = normals.begin(); itNormals != normals.end() ; itNormals++)
        (*itNormals).second.first.normalize();

    // compute coordinates of extruded vertices (including initial vertices)
    for (unsigned int i=0; i<surfaceVertices.size(); i++)
    {
        for (int n=0; n<=f_numberOfSlices.getValue(); n++)
        {
            Real scale = (f_thicknessIn.getValue() + f_thicknessOut.getValue())/(Real)f_numberOfSlices.getValue();
            Coord disp = -normals[i].first * f_thicknessIn.getValue() + (normals[i].first * scale * (Real)n);
            Coord newVertexPos(surfaceVertices[i][0]*f_scale.getValue()[0], surfaceVertices[i][1]*f_scale.getValue()[1], surfaceVertices[i][2]*f_scale.getValue()[2]);
            extrudedVertices->push_back(newVertexPos + disp);
        }
    }

    // compute indices of newly created surface quads
    for (unsigned int q=0; q<surfaceQuads.size(); q++)
    {
        BaseMeshTopology::Quad quad = BaseMeshTopology::Quad(surfaceQuads[q][0]*(nSlices+1), surfaceQuads[q][1]*(nSlices+1), surfaceQuads[q][2]*(nSlices+1), surfaceQuads[q][3]*(nSlices+1));
        extrudedSurfaceQuads->push_back(quad);
    }

    // compute indices of extruded quads (including initial quads)
    for (unsigned int q=0; q<surfaceQuads.size(); q++)
    {
        BaseMeshTopology::Quad quad;

        // quad on the outer surface
        quad = BaseMeshTopology::Quad(surfaceQuads[q][0]*(nSlices+1), surfaceQuads[q][1]*(nSlices+1), surfaceQuads[q][2]*(nSlices+1), surfaceQuads[q][3]*(nSlices+1));
        extrudedQuads->push_back(quad);
        // quads on the inner surface
        quad = BaseMeshTopology::Quad(surfaceQuads[q][3]*(nSlices+1)+nSlices, surfaceQuads[q][2]*(nSlices+1)+nSlices, surfaceQuads[q][1]*(nSlices+1)+nSlices, surfaceQuads[q][0]*(nSlices+1)+nSlices);
        extrudedQuads->push_back(quad);
        // intermediate quads parallel to the surface
        for (int n=1; n<nSlices; n++)
        {
            quad = BaseMeshTopology::Quad(surfaceQuads[q][0]*(nSlices+1)+n, surfaceQuads[q][1]*(nSlices+1)+n, surfaceQuads[q][2]*(nSlices+1)+n, surfaceQuads[q][3]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
            quad = BaseMeshTopology::Quad(surfaceQuads[q][3]*(nSlices+1)+n, surfaceQuads[q][2]*(nSlices+1)+n, surfaceQuads[q][1]*(nSlices+1)+n, surfaceQuads[q][0]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
        }
        // intermediate quads "orthogonal" to the surface
        for (int n=0; n<=nSlices-1; n++)
        {
            quad = BaseMeshTopology::Quad(surfaceQuads[q][0]*(nSlices+1)+n, surfaceQuads[q][0]*(nSlices+1)+n+1, surfaceQuads[q][1]*(nSlices+1)+n+1, surfaceQuads[q][1]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
            quad = BaseMeshTopology::Quad(surfaceQuads[q][1]*(nSlices+1)+n, surfaceQuads[q][1]*(nSlices+1)+n+1, surfaceQuads[q][2]*(nSlices+1)+n+1, surfaceQuads[q][2]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
            quad = BaseMeshTopology::Quad(surfaceQuads[q][2]*(nSlices+1)+n, surfaceQuads[q][2]*(nSlices+1)+n+1, surfaceQuads[q][3]*(nSlices+1)+n+1, surfaceQuads[q][3]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
            quad = BaseMeshTopology::Quad(surfaceQuads[q][3]*(nSlices+1)+n, surfaceQuads[q][3]*(nSlices+1)+n+1, surfaceQuads[q][0]*(nSlices+1)+n+1, surfaceQuads[q][0]*(nSlices+1)+n);
            extrudedQuads->push_back(quad);
        }
    }

    // compute indices of newly created hexas
    for (unsigned int q=0; q<surfaceQuads.size(); q++)
    {
        for (int n=0; n<nSlices; n++)
        {
            BaseMeshTopology::Hexa hexa = BaseMeshTopology::Hexa(surfaceQuads[q][3]*(nSlices+1)+n,   surfaceQuads[q][2]*(nSlices+1)+n,   surfaceQuads[q][1]*(nSlices+1)+n,   surfaceQuads[q][0]*(nSlices+1)+n,
                    surfaceQuads[q][3]*(nSlices+1)+n+1, surfaceQuads[q][2]*(nSlices+1)+n+1, surfaceQuads[q][1]*(nSlices+1)+n+1, surfaceQuads[q][0]*(nSlices+1)+n+1);
            extrudedHexas->push_back(hexa);
        }
    }

    f_extrudedHexas.endEdit();
    f_extrudedQuads.endEdit();
    f_extrudedSurfaceQuads.endEdit();
    f_extrudedVertices.endEdit();
}


template <class DataTypes>
void ExtrudeQuadsAndGenerateHexas<DataTypes>::draw(const core::visual::VisualParams *)
{

}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
