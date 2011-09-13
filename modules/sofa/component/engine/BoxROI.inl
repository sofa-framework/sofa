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
#ifndef SOFA_COMPONENT_ENGINE_BOXROI_INL
#define SOFA_COMPONENT_ENGINE_BOXROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/BoxROI.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/VisualParams.h>

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
BoxROI<DataTypes>::BoxROI()
    : boxes( initData(&boxes, "box", "Box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , f_edges(initData (&f_edges, "edges", "Edge Topology") )
    , f_triangles(initData (&f_triangles, "triangles", "Triangle Topology") )
    , f_tetrahedra(initData (&f_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , f_computeEdges( initData(&f_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI.") )
    , f_computeTriangles( initData(&f_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI.") )
    , f_computeTetrahedra( initData(&f_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI.") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_edgeIndices( initData(&f_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , f_triangleIndices( initData(&f_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , f_tetrahedronIndices( initData(&f_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , f_edgesInROI( initData(&f_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , f_trianglesInROI( initData(&f_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , f_tetrahedraInROI( initData(&f_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , p_drawBoxes( initData(&p_drawBoxes,false,"drawBoxes","Draw Box(es)") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , p_drawEdges( initData(&p_drawEdges,false,"drawEdges","Draw Edges") )
    , p_drawTriangles( initData(&p_drawTriangles,false,"drawTriangles","Draw Triangles") )
    , p_drawTetrahedra( initData(&p_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","rendering size for box and topological elements") )
{
    //Adding alias to handle old BoxROI input/output
    addAlias(&f_pointsInROI,"pointsInBox");
    addAlias(&f_edgesInROI,"edgesInBox");
    addAlias(&f_trianglesInROI,"f_trianglesInBox");
    addAlias(&f_tetrahedraInROI,"f_tetrahedraInBox");
    addAlias(&f_X0,"rest_position");

    //Adding alias to handle TrianglesInBoxROI input/output
    addAlias(&p_drawBoxes,"isVisible");

    boxes.beginEdit()->push_back(Vec6(0,0,0,1,1,1));
    boxes.endEdit();

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void BoxROI<DataTypes>::init()
{
    //cerr<<"BoxROI<DataTypes>::init() is called "<<endl;
    if (!f_X0.isSet())
    {
        //cerr<<"BoxROI<DataTypes>::init() f_X0 is not set "<<endl;
        BaseMechanicalState* mstate;
        this->getContext()->get(mstate);
        if (mstate)
        {
            BaseData* parent = mstate->findField("rest_position");
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
            if (!f_edges.isSet() && f_computeEdges.getValue())
            {
                BaseData* eparent = topology->findField("edges");
                if (eparent)
                {
                    f_edges.setParent(eparent);
                    f_edges.setReadOnly(true);
                }
            }
            if (!f_triangles.isSet() && f_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findField("triangles");
                if (tparent)
                {
                    f_triangles.setParent(tparent);
                    f_triangles.setReadOnly(true);
                }
            }
            if (!f_tetrahedra.isSet() && f_computeTetrahedra.getValue())
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
    addOutput(&f_pointsInROI);
    addOutput(&f_edgesInROI);
    addOutput(&f_trianglesInROI);
    addOutput(&f_tetrahedraInROI);
    setDirtyValue();

    //cerr<<"BoxROI<DataTypes>::init() -> f_X0 = "<<f_X0<<endl;
    //cerr<<"BoxROI<DataTypes>::init() -> boxes = "<<boxes<<endl;
    //cerr<<"BoxROI<DataTypes>::init() -> f_indices = "<<f_indices<<endl;
}

template <class DataTypes>
void BoxROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInBox(const typename DataTypes::CPos& p, const Vec6& b)
{
    return ( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] );
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInBox(const PointID& pid, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( isPointInBox(p,b) );
}

template <class DataTypes>
bool BoxROI<DataTypes>::isEdgeInBox(const Edge& e, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInBox(c,b);
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTriangleInBox(const Triangle& t, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInBox(c,b));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTetrahedronInBox(const Tetra &t, const Vec6 &b)
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
void BoxROI<DataTypes>::update()
{
    cleanDirty();

    helper::vector<Vec6>& vb = *(boxes.beginEdit());

    if (vb.empty())
    {
        boxes.endEdit();
        return;
    }

    for (unsigned int bi=0; bi<vb.size(); ++bi)
    {
        if (vb[bi][0] > vb[bi][3]) std::swap(vb[bi][0],vb[bi][3]);
        if (vb[bi][1] > vb[bi][4]) std::swap(vb[bi][1],vb[bi][4]);
        if (vb[bi][2] > vb[bi][5]) std::swap(vb[bi][2],vb[bi][5]);
    }

    boxes.endEdit();

    // Read accessor for input topology
    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;

    // Write accessor for topological element indices in BOX
    SetIndex& indices = *f_indices.beginEdit();
    SetIndex& edgeIndices = *f_edgeIndices.beginEdit();
    SetIndex& triangleIndices = *f_triangleIndices.beginEdit();
    SetIndex& tetrahedronIndices = *f_tetrahedronIndices.beginEdit();

    // Write accessor for toplogical element in BOX
    helper::WriteAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
    helper::WriteAccessor< Data<helper::vector<Edge> > > edgesInROI = f_edgesInROI;
    helper::WriteAccessor< Data<helper::vector<Triangle> > > trianglesInROI = f_trianglesInROI;
    helper::WriteAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;

    // Clear lists
    indices.clear();
    edgeIndices.clear();
    triangleIndices.clear();
    tetrahedronIndices.clear();

    pointsInROI.clear();
    edgesInROI.clear();
    trianglesInROI.clear();
    tetrahedraInROI.clear();


    const VecCoord* x0 = &f_X0.getValue();

    //Points
    for( unsigned i=0; i<x0->size(); ++i )
    {
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isPointInBox(i, vb[bi]))
            {
                indices.push_back(i);
                pointsInROI.push_back((*x0)[i]);
//                sout<<"\nBoxROI<DataTypes>::update, add index "<< i << sendl;
                break;
            }
        }
    }

    //Edges
    if (f_computeEdges.getValue())
    {
        for(unsigned int i=0 ; i<edges.size() ; i++)
        {
            Edge e = edges[i];
            for (unsigned int bi=0; bi<vb.size(); ++bi)
            {
                if (isEdgeInBox(e, vb[bi]))
                {
                    edgeIndices.push_back(i);
                    edgesInROI.push_back(e);
                    break;
                }
            }
        }
    }

    //Triangles
    if (f_computeTriangles.getValue())
    {
        for(unsigned int i=0 ; i<triangles.size() ; i++)
        {
            Triangle t = triangles[i];
            for (unsigned int bi=0; bi<vb.size(); ++bi)
            {
                if (isTriangleInBox(t, vb[bi]))
                {
                    triangleIndices.push_back(i);
                    trianglesInROI.push_back(t);
                    break;
                }
            }
        }
    }

    //Tetrahedra
    if (f_computeTetrahedra.getValue())
    {
        for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
        {
            Tetra t = tetrahedra[i];
            for (unsigned int bi=0; bi<vb.size(); ++bi)
            {
                if (isTetrahedronInBox(t, vb[bi]))
                {
                    tetrahedronIndices.push_back(i);
                    tetrahedraInROI.push_back(t);
                    break;
                }
            }
        }
    }

    f_indices.endEdit();
    f_edgeIndices.endEdit();
    f_triangleIndices.endEdit();
    f_tetrahedronIndices.endEdit();
}

template <class DataTypes>
void BoxROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() && !this->_drawSize.getValue())
        return;

    const VecCoord* x0 = &f_X0.getValue();
    glColor3f(1.0f, 0.4f, 0.4f);

    ///draw the boxes
    if( p_drawBoxes.getValue())
    {
        glDisable(GL_LIGHTING);
        if (_drawSize.getValue())
            glLineWidth((GLfloat)_drawSize.getValue());
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
        glLineWidth(1);
    }

    ///draw points in ROI
    if( p_drawPoints.getValue())
    {
        if (_drawSize.getValue())
            glPointSize((GLfloat)_drawSize.getValue());
        glDisable(GL_LIGHTING);
        glBegin(GL_POINTS);
        glPointSize(5.0);
        helper::ReadAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            helper::gl::glVertexT(p);
        }
        glEnd();
        glPointSize(1);
    }

    ///draw edges in ROI
    if( p_drawEdges.getValue())
    {
        glDisable(GL_LIGHTING);
        glLineWidth((GLfloat)_drawSize.getValue());
        glBegin(GL_LINES);
        helper::ReadAccessor< Data<helper::vector<Edge> > > edgesInROI = f_edgesInROI;
        for (unsigned int i=0; i<edgesInROI.size() ; ++i)
        {
            Edge e = edgesInROI[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                CPos p = DataTypes::getCPos((*x0)[e[j]]);
                helper::gl::glVertexT(p);
            }
        }
        glEnd();
        glLineWidth(1);
    }

    ///draw triangles in ROI
    if( p_drawTriangles.getValue())
    {
        glDisable(GL_LIGHTING);
        glBegin(GL_TRIANGLES);
        helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesInROI = f_trianglesInROI;
        for (unsigned int i=0; i<trianglesInROI.size() ; ++i)
        {
            Triangle t = trianglesInROI[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                CPos p = DataTypes::getCPos((*x0)[t[j]]);
                helper::gl::glVertexT(p);
            }
        }
        glEnd();
    }

    ///draw tetrahedra in ROI
    if( p_drawTetrahedra.getValue())
    {
        glDisable(GL_LIGHTING);
        glLineWidth((GLfloat)_drawSize.getValue());
        glBegin(GL_LINES);
        helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;
        for (unsigned int i=0; i<tetrahedraInROI.size() ; ++i)
        {
            Tetra t = tetrahedraInROI[i];
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
        glLineWidth(1);
    }
}


template <class DataTypes>
bool BoxROI<DataTypes>::addBBox(double* minBBox, double* maxBBox)
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
