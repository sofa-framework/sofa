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
#ifndef SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_INL
#define SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/ExtrudeSurface.h>
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
ExtrudeSurface<DataTypes>::ExtrudeSurface()
    : initialized(false)
    , isVisible( initData (&isVisible, bool (true), "isVisible", "is Visible ?") )
    , heightFactor( initData (&heightFactor, Real (1.0), "heightFactor", "Factor for the height of the extrusion (based on normal) ?") )
    , f_triangles(initData(&f_triangles, "triangles", "List of triangle indices"))
    , f_extrusionVertices( initData (&f_extrusionVertices, "extrusionVertices", "Position coordinates of the extrusion") )
    , f_surfaceVertices( initData (&f_surfaceVertices, "surfaceVertices", "Position coordinates of the surface") )
    , f_extrusionTriangles( initData (&f_extrusionTriangles, "extrusionTriangles", "Triangles indices of the extrusion") )
    , f_surfaceTriangles( initData (&f_surfaceTriangles, "surfaceTriangles", "Indices of the triangles of the surface to extrude") )
{
}

template <class DataTypes>
void ExtrudeSurface<DataTypes>::init()
{
    addInput(&f_surfaceTriangles);
    addInput(&f_surfaceVertices);
    addInput(&f_triangles);

    addOutput(&f_extrusionVertices);
    addOutput(&f_extrusionTriangles);
    setDirtyValue();
}

template <class DataTypes>
void ExtrudeSurface<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ExtrudeSurface<DataTypes>::update()
{
    using sofa::core::topology::BaseMeshTopology;


    const helper::vector<BaseMeshTopology::TriangleID>& surfaceTriangles = f_surfaceTriangles.getValue();
    const VecCoord& surfaceVertices = f_surfaceVertices.getValue();

    if (surfaceVertices.size() <= 1 && surfaceTriangles.size() <= 1)
        return;

    const BaseMeshTopology::SeqTriangles* triangles = &f_triangles.getValue();

    cleanDirty();

    VecCoord* extrusionVertices = f_extrusionVertices.beginWriteOnly();
    extrusionVertices->clear();
    helper::vector<BaseMeshTopology::Triangle>* extrusionTriangles = f_extrusionTriangles.beginWriteOnly();
    extrusionTriangles->clear();

    helper::vector<BaseMeshTopology::TriangleID>::const_iterator itTriangles, itTrianglesSide;

    std::map<int, int> pointMatching;
    std::map<BaseMeshTopology::Edge, bool > edgesOnBorder;
    std::set<int> pointsUsed;
    std::map<int, std::pair<Vec3, unsigned int> > normals;
    //first loop to compute normals per point
    for (itTriangles=surfaceTriangles.begin() ; itTriangles != surfaceTriangles.end() ; itTriangles++)
    {
        BaseMeshTopology::Triangle triangle = (*triangles)[(*itTriangles)];
        VecCoord triangleCoord;

        //fetch real coords
        for (unsigned int i=0 ; i<3 ; i++)
            triangleCoord.push_back(surfaceVertices[triangle[i]]);

        //compute normal
        Coord n =  cross(triangleCoord[1]-triangleCoord[0], triangleCoord[2]-triangleCoord[0]);
        n.normalize();
        for (unsigned int i=0 ; i<3 ; i++)
        {
            normals[triangle[i]].first += n;
            normals[triangle[i]].second++;
        }
    }
    //average normals
    typename std::map<int, std::pair<Vec3, unsigned int> >::iterator itNormals;
    for (itNormals = normals.begin(); itNormals != normals.end() ; itNormals++)
    {
        //(*itNormals).second.first /= (*itNormals).second.second;
        (*itNormals).second.first.normalize();
    }

    for (itTriangles=surfaceTriangles.begin() ; itTriangles != surfaceTriangles.end() ; itTriangles++)
    {
        BaseMeshTopology::Triangle triangle = (*triangles)[(*itTriangles)];
        BaseMeshTopology::Triangle t1, t2;

        //create triangle from surface and the new triangle
        //vertex created from surface has an even (2*n) index
        //vertex created from the addition with the normal has an odd (2*n + 1) index
        //a table is also used to map old vertex indices with the new set of indices
        for (unsigned int i=0 ; i<3 ; i++)
        {
            if (pointMatching.find(triangle[i]) == pointMatching.end())
            {
                extrusionVertices->push_back(surfaceVertices[triangle[i]]);
                extrusionVertices->push_back(surfaceVertices[triangle[i]] + normals[triangle[i]].first*heightFactor.getValue());

                pointMatching[triangle[i]] = extrusionVertices->size() - 2;

                t1[i] = extrusionVertices->size()-2;
                t2[i] = extrusionVertices->size()-1;
            }
            else
            {
                t1[i] = pointMatching[triangle[i]];
                t2[i] = pointMatching[triangle[i]] + 1;
            }
        }

        //to get borders, we simply stock the edge and look if it is already in the table
        BaseMeshTopology::Edge e[3];
        BaseMeshTopology::Edge ei[3];

        { e[0][0] = t1[0] ; e[0][1] = t1[1] ; }
        { ei[0][0] = t1[1] ; ei[0][1] = t1[0] ; }

        { e[1][0] = t1[0] ; e[1][1] = t1[2] ; }
        { ei[1][0] = t1[2] ; ei[1][1] = t1[0] ; }

        { e[2][0] = t1[1] ; e[2][1] = t1[2] ; }
        { ei[2][1] = t1[1] ; ei[2][0] = t1[2] ; }

        for (unsigned int i=0 ; i<3 ; i++)
        {
            if ( edgesOnBorder.find(e[i])  == edgesOnBorder.end())
            {
                if ( edgesOnBorder.find(ei[i])  == edgesOnBorder.end())
                    edgesOnBorder[e[i]] = true;
                else
                    edgesOnBorder[ei[i]] = false;
            }
            else
                edgesOnBorder[e[i]] = false;

        }

        //flip normal
        std::swap(t1[1], t1[2]);

        extrusionTriangles->push_back(t1);
        extrusionTriangles->push_back(t2);
    }

    std::map<BaseMeshTopology::Edge, bool >::const_iterator itEdges;
    for (itEdges = edgesOnBorder.begin() ; itEdges != edgesOnBorder.end() ; itEdges++)
    {
        //for each edge, we can get the "mirrored one" and construct 2 other triangles
        if ((*itEdges).second)
        {
            BaseMeshTopology::Edge e = (*itEdges).first;
            BaseMeshTopology::Triangle ft1, ft2;

            //first triangle
            ft1[0] = e[0];
            ft1[1] = e[1];
            ft1[2] = e[0] + 1;

            //second triangle
            ft2[0] = e[1] + 1;
            ft2[1] = e[0] + 1;
            ft2[2] = e[1];

            extrusionTriangles->push_back(ft1);
            extrusionTriangles->push_back(ft2);
        }
    }

    f_extrusionTriangles.endEdit();
    f_extrusionVertices.endEdit();
}

template <class DataTypes>
void ExtrudeSurface<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL

    using sofa::core::topology::BaseMeshTopology;

    const helper::vector<BaseMeshTopology::TriangleID> &surfaceTriangles = f_surfaceTriangles.getValue();

    helper::vector<BaseMeshTopology::TriangleID>::const_iterator itTriangles;
    glDisable(GL_LIGHTING);

    if (!vparams->displayFlags().getShowBehaviorModels() || !isVisible.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const helper::vector<BaseMeshTopology::Triangle> &extrusionTriangles = f_extrusionTriangles.getValue();
    const VecCoord& extrusionVertices = f_extrusionVertices.getValue();
    helper::vector<BaseMeshTopology::Triangle>::const_iterator it;

    //Triangles From Surface

    glColor3f(1.0,0.0,0.0);
    glBegin(GL_TRIANGLES);
    for (unsigned int i=0 ; i<surfaceTriangles.size()*2 ; i+=2)
    {
        BaseMeshTopology::Triangle triangle = extrusionTriangles[i];

        for (unsigned int j=0 ; j<3 ; j++)
        {
            Coord p = (extrusionVertices[triangle[j]]);
            glVertex3d(p[0], p[1], p[2]);
        }
    }
    glEnd();

    //Triangles From Extrusion
    glColor3f(0.0,1.0,0.0);
    glBegin(GL_TRIANGLES);
    for (unsigned int i=1 ; i<surfaceTriangles.size()*2 ; i+=2)
    {
        BaseMeshTopology::Triangle triangle = extrusionTriangles[i];

        for (unsigned int j=0 ; j<3 ; j++)
        {
            Coord p = (extrusionVertices[triangle[j]]);
            glVertex3d(p[0], p[1], p[2]);
        }
    }
    glEnd();

    //Border Triangles
    glColor3f(0.0,0.0,1.0);
    glBegin(GL_TRIANGLES);
    for (unsigned int i=surfaceTriangles.size()*2 ; i<extrusionTriangles.size() ; i++)
    {
        BaseMeshTopology::Triangle triangle = extrusionTriangles[i];

        for (unsigned int j=0 ; j<3 ; j++)
        {
            Coord p = (extrusionVertices[triangle[j]]);
            glVertex3d(p[0], p[1], p[2]);
        }
    }
    glEnd();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHTING);
#endif /* SOFA_NO_OPENGL */
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
