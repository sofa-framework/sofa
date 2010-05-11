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
#ifndef SOFA_COMPONENT_ENGINE_SUBSETTOPOLOGY_INL
#define SOFA_COMPONENT_ENGINE_SUBSETTOPOLOGY_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/SubsetTopology.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;
using namespace core::topology;

template <class DataTypes>
SubsetTopology<DataTypes>::SubsetTopology()
    : boxes( initData(&boxes, "box", "Box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_X0( initData (&f_X0, "rest_position", "Rest position coordinates of the degrees of freedom") )
    , f_edges(initData (&f_edges, "edges", "Edge Topology") )
    , f_triangles(initData (&f_triangles, "triangles", "Triangle Topology") )
    , f_tetrahedra(initData (&f_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_edgeIndices( initData(&f_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , f_triangleIndices( initData(&f_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , f_tetrahedronIndices( initData(&f_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , f_pointsInBox( initData(&f_pointsInBox,"pointsInBox","Points contained in the ROI") )
    , f_pointsOutBox( initData(&f_pointsOutBox,"pointsOutBox","Points out of the ROI") )
    , f_edgesInBox( initData(&f_edgesInBox,"edgesInBox","Edges contained in the ROI") )
    , f_trianglesInBox( initData(&f_trianglesInBox,"f_trianglesInBox","Triangles contained in the ROI") )
    , f_trianglesOutBox( initData(&f_trianglesOutBox,"f_trianglesOutBox","Triangles out of the ROI") )
    , f_tetrahedraInBox( initData(&f_tetrahedraInBox,"f_tetrahedraInBox","Tetrahedra contained in the ROI") )
    , f_tetrahedraOutBox( initData(&f_tetrahedraOutBox,"f_tetrahedraOutBox","Tetrahedra out of the ROI") )
    , p_drawBoxes( initData(&p_drawBoxes,false,"drawBoxes","Draw Box(es)") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , p_drawEdges( initData(&p_drawEdges,false,"drawEdges","Draw Edges") )
    , p_drawTriangles( initData(&p_drawTriangles,false,"drawTriangle","Draw Triangles") )
    , p_drawTetrahedra( initData(&p_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra") )
{
    boxes.beginEdit()->push_back(Vec6(0,0,0,1,1,1));
    boxes.endEdit();

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void SubsetTopology<DataTypes>::init()
{

    if (!f_X0.isSet())
    {
        MechanicalState<DataTypes>* mstate;
        this->getContext()->get(mstate);
        if (mstate)
        {
            BaseData* parent = mstate->findField("position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader);
            if (loader)
            {
                BaseData* parent = loader->findField("position");
                if (parent)
                {
                    f_X0.setParent(parent);
                    f_X0.setReadOnly(true);
                }
            }
        }
    }
    if (!f_edges.isSet() || !f_triangles.isSet() || !f_tetrahedra.isSet())
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology);
        if (topology)
        {
            if (!f_edges.isSet())
            {
                BaseData* eparent = topology->findField("edges");
                if (eparent)
                {
                    f_edges.setParent(eparent);
                    f_edges.setReadOnly(true);
                }
            }
            if (!f_triangles.isSet())
            {
                BaseData* tparent = topology->findField("triangles");
                if (tparent)
                {
                    f_triangles.setParent(tparent);
                    f_triangles.setReadOnly(true);
                }
            }
            if (!f_tetrahedra.isSet())
            {
                BaseData* tparent = topology->findField("tetrahedra");
                if (tparent)
                {
                    f_tetrahedra.setParent(tparent);
                    f_tetrahedra.setReadOnly(true);
                }
            }
        }
    }

    addInput(&f_X0);
    addInput(&f_edges);
    addInput(&f_triangles);
    addInput(&f_tetrahedra);

    addOutput(&f_indices);
    addOutput(&f_edgeIndices);
    addOutput(&f_triangleIndices);
    addOutput(&f_tetrahedronIndices);
    addOutput(&f_pointsInBox);
    addOutput(&f_pointsOutBox);
    addOutput(&f_edgesInBox);
    addOutput(&f_trianglesInBox);
    addOutput(&f_trianglesOutBox);
    addOutput(&f_tetrahedraInBox);
    addOutput(&f_tetrahedraOutBox);
    setDirtyValue();
}

template <class DataTypes>
void SubsetTopology<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::isPointInBox(const typename DataTypes::CPos& p, const Vec6& b)
{
    return ( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] );
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::isPointInBox(const PointID& pid, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( isPointInBox(p,b) );
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::isEdgeInBox(const Edge& e, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInBox(c,b);
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::isTriangleInBox(const Triangle& t, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInBox(c,b));
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::isTetrahedronInBox(const Tetra &t, const Vec6 &b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBox(c,b));
}

template <class DataTypes>
void SubsetTopology<DataTypes>::update()
{
    cleanDirty();

    helper::vector<Vec6>& vb = *(boxes.beginEdit());

    for (unsigned int bi=0; bi<vb.size(); ++bi)
    {
        if (vb[bi][0] > vb[bi][3]) std::swap(vb[bi][0],vb[bi][3]);
        if (vb[bi][1] > vb[bi][4]) std::swap(vb[bi][1],vb[bi][4]);
        if (vb[bi][2] > vb[bi][5]) std::swap(vb[bi][2],vb[bi][5]);
    }

    boxes.endEdit();

    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;

    SetIndex& indices = *f_indices.beginEdit();
    SetIndex& edgeIndices = *f_edgeIndices.beginEdit();
    SetIndex& triangleIndices = *f_triangleIndices.beginEdit();
    SetIndex& tetrahedronIndices = *f_tetrahedronIndices.beginEdit();

    helper::WriteAccessor< Data<VecCoord > > pointsInBox = f_pointsInBox;
    helper::WriteAccessor< Data<VecCoord > > pointsOutBox = f_pointsOutBox;
    helper::WriteAccessor< Data<helper::vector<Edge> > > edgesInBox = f_edgesInBox;
    helper::WriteAccessor< Data<helper::vector<Triangle> > > trianglesInBox = f_trianglesInBox;
    helper::WriteAccessor< Data<helper::vector<Triangle> > > trianglesOutBox = f_trianglesOutBox;

    helper::WriteAccessor< Data<helper::vector<Tetra> > > tetrahedraInBox = f_tetrahedraInBox;
    helper::WriteAccessor< Data<helper::vector<Tetra> > > tetrahedraOutBox = f_tetrahedraOutBox;

    indices.clear();
    edgesInBox.clear();
    trianglesInBox.clear();
    trianglesOutBox.clear();
    tetrahedraInBox.clear();
    tetrahedraOutBox.clear();
    pointsOutBox.clear();
    pointsInBox.clear();


    const VecCoord* x0 = &f_X0.getValue();

    for( unsigned i=0; i<x0->size(); ++i )
    {
        bool inside = false;
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isPointInBox(i, vb[bi]))
            {
                indices.push_back(i);
                pointsInBox.push_back((*x0)[i]);
                inside = true;
                break;
            }
        }

        if (!inside)
            pointsOutBox.push_back((*x0)[i]);
    }

    for(unsigned int i=0 ; i<edges.size() ; i++)
    {
        Edge e = edges[i];
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isEdgeInBox(e, vb[bi]))
            {
                edgeIndices.push_back(i);
                edgesInBox.push_back(e);
                break;
            }
        }
    }

    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        Triangle t = triangles[i];
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isTriangleInBox(t, vb[bi]))
            {
                triangleIndices.push_back(i);
                trianglesInBox.push_back(t);
                break;
            }
            else
                trianglesOutBox.push_back(t);
        }
    }

    for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
    {
        Tetra t = tetrahedra[i];
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isTetrahedronInBox(t, vb[bi]))
            {
                tetrahedronIndices.push_back(i);
                tetrahedraInBox.push_back(t);
                break;
            }
            else
                tetrahedraOutBox.push_back(t);
        }
    }

    f_indices.endEdit();
    f_edgeIndices.endEdit();
    f_triangleIndices.endEdit();
    f_tetrahedronIndices.endEdit();

}

template <class DataTypes>
void SubsetTopology<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels())
        return;

    const VecCoord* x0 = &f_X0.getValue();
    glColor3f(0.0, 1.0, 1.0);
    if( p_drawBoxes.getValue())
    {
        ///draw the boxes
        glBegin(GL_LINES);
        const helper::vector<Vec6>& vb=boxes.getValue();
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            const Vec6& b=vb[bi];
            const Real& Xmin=b[0];
            const Real& Xmax=b[3];
            const Real& Ymin=b[1];
            const Real& Ymax=b[4];
            const Real& Zmin=b[2];
            const Real& Zmax=b[5];
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymax,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmax,Ymax,Zmax);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmax,Ymax,Zmax);
        }
        glEnd();
    }
    if( p_drawPoints.getValue())
    {
        ///draw points in boxes
        glBegin(GL_POINTS);
        glPointSize(5.0);
        helper::ReadAccessor< Data<VecCoord > > pointsInBox = f_pointsInBox;
        for (unsigned int i=0; i<pointsInBox.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInBox[i]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
    if( p_drawEdges.getValue())
    {
        ///draw edges in boxes
        glBegin(GL_LINES);
        helper::ReadAccessor< Data<helper::vector<Edge> > > edgesInBox = f_edgesInBox;
        for (unsigned int i=0; i<edgesInBox.size() ; ++i)
        {
            Edge e = edgesInBox[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                CPos p = DataTypes::getCPos((*x0)[e[j]]);
                helper::gl::glVertexT(p);
            }
        }
        glEnd();
    }
    if( p_drawTriangles.getValue())
    {
        ///draw triangles in boxes
        glBegin(GL_TRIANGLES);
        helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesInBox = f_trianglesInBox;
        for (unsigned int i=0; i<trianglesInBox.size() ; ++i)
        {
            Triangle t = trianglesInBox[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                CPos p = DataTypes::getCPos((*x0)[t[j]]);
                helper::gl::glVertexT(p);
            }
        }
        glEnd();
    }

    if( p_drawTetrahedra.getValue())
    {
        ///draw tetrahedra in boxes
        glBegin(GL_LINES);
        helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraInBox = f_tetrahedraInBox;
        for (unsigned int i=0; i<tetrahedraInBox.size() ; ++i)
        {
            Tetra t = tetrahedraInBox[i];
            for (unsigned int j=0 ; j<4 ; j++)
            {
                CPos p = DataTypes::getCPos((*x0)[t[j]]);
                helper::gl::glVertexT(p);
                p = DataTypes::getCPos((*x0)[t[(j+1)%4]]);
                helper::gl::glVertexT(p);
            }

            CPos p = DataTypes::getCPos((*x0)[t[0]]);
            helper::gl::glVertexT(p);
            p = DataTypes::getCPos((*x0)[t[2]]);
            helper::gl::glVertexT(p);
            p = DataTypes::getCPos((*x0)[t[1]]);
            helper::gl::glVertexT(p);
            p = DataTypes::getCPos((*x0)[t[3]]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
}

template <class DataTypes>
bool SubsetTopology<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const helper::vector<Vec6>& vb=boxes.getValue();
    for (unsigned int bi=0; bi<vb.size(); ++bi)
    {
        const Vec6& b=vb[bi];
        if (b[0] < minBBox[0]) minBBox[0] = b[0];
        if (b[1] < minBBox[1]) minBBox[1] = b[1];
        if (b[2] < minBBox[2]) minBBox[2] = b[2];
        if (b[3] > maxBBox[0]) maxBBox[0] = b[3];
        if (b[4] > maxBBox[1]) maxBBox[1] = b[4];
        if (b[5] > maxBBox[2]) maxBBox[2] = b[5];
    }
    return true;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
