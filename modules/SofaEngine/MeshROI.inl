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
#ifndef SOFA_COMPONENT_ENGINE_MESHROI_INL
#define SOFA_COMPONENT_ENGINE_MESHROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaEngine/MeshROI.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
MeshROI<DataTypes>::MeshROI()
    : f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , f_edges(initData (&f_edges, "edges", "Edge Topology") )
    , f_triangles(initData (&f_triangles, "triangles", "Triangle Topology") )
    , f_tetrahedra(initData (&f_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , f_X0_i( initData (&f_X0_i, "ROIposition", "ROI position coordinates of the degrees of freedom") )
    , f_edges_i(initData (&f_edges_i, "ROIedges", "ROI Edge Topology") )
    , f_triangles_i(initData (&f_triangles_i, "ROItriangles", "ROI Triangle Topology") )
    , f_computeEdges( initData(&f_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI.") )
    , f_computeTriangles( initData(&f_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI.") )
    , f_computeTetrahedra( initData(&f_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI.") )
    , f_computeTemplateTriangles( initData(&f_computeTemplateTriangles,true,"computeMeshROI","Compute with the mesh (not only bounding box)") )
    , f_box( initData(&f_box, "box", "Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_edgeIndices( initData(&f_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , f_triangleIndices( initData(&f_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , f_tetrahedronIndices( initData(&f_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , f_edgesInROI( initData(&f_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , f_trianglesInROI( initData(&f_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , f_tetrahedraInROI( initData(&f_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , f_pointsOutROI( initData(&f_pointsOutROI,"pointsOutROI","Points not contained in the ROI") )
    , f_edgesOutROI( initData(&f_edgesOutROI,"edgesOutROI","Edges not contained in the ROI") )
    , f_trianglesOutROI( initData(&f_trianglesOutROI,"trianglesOutROI","Triangles not contained in the ROI") )
    , f_tetrahedraOutROI( initData(&f_tetrahedraOutROI,"tetrahedraOutROI","Tetrahedra not contained in the ROI") )
    , f_indicesOut( initData(&f_indicesOut,"indicesOut","Indices of the points not contained in the ROI") )
    , f_edgeOutIndices( initData(&f_edgeOutIndices,"edgeOutIndices","Indices of the edges not contained in the ROI") )
    , f_triangleOutIndices( initData(&f_triangleOutIndices,"triangleOutIndices","Indices of the triangles not contained in the ROI") )
    , f_tetrahedronOutIndices( initData(&f_tetrahedronOutIndices,"tetrahedronOutIndices","Indices of the tetrahedra not contained in the ROI") )
    , p_drawOut( initData(&p_drawOut,false,"drawOut","Draw the data not contained in the ROI") )
    , p_drawMesh( initData(&p_drawMesh,false,"drawMesh","Draw Mesh used for the ROI") )
    , p_drawBox( initData(&p_drawBox,false,"drawBox","Draw the Bounding box around the mesh used for the ROI") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , p_drawEdges( initData(&p_drawEdges,false,"drawEdges","Draw Edges") )
    , p_drawTriangles( initData(&p_drawTriangles,false,"drawTriangles","Draw Triangles") )
    , p_drawTetrahedra( initData(&p_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","rendering size for mesh and topological elements") )
    , p_doUpdate( initData(&p_doUpdate,false,"doUpdate","Update the computation (not only at the init") )
{
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void MeshROI<DataTypes>::init()
{
    using sofa::core::objectmodel::BaseData;
    using sofa::core::objectmodel::BaseContext;
    using sofa::core::topology::BaseMeshTopology;
    using sofa::core::behavior::BaseMechanicalState;

    //cerr<<"MeshROI<DataTypes>::init() is called "<<endl;
    if (!f_X0.isSet())
    {
        //cerr<<"MeshROI<DataTypes>::init() f_X0 is not set "<<endl;
        sofa::core::behavior::BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader,BaseContext::Local); // perso
            if (loader)
            {
                BaseData* parent = loader->findData("position");
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
        this->getContext()->get(topology,BaseContext::Local); // perso
        if (topology)
        {
            if (!f_edges.isSet() && f_computeEdges.getValue())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    f_edges.setParent(eparent);
                    f_edges.setReadOnly(true);
                }
            }
            if (!f_triangles.isSet() && f_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    f_triangles.setParent(tparent);
                    f_triangles.setReadOnly(true);
                }
            }
            if (!f_tetrahedra.isSet() && f_computeTetrahedra.getValue())
            {
                BaseData* tparent = topology->findData("tetrahedra");
                if (tparent)
                {
                    f_tetrahedra.setParent(tparent);
                    f_tetrahedra.setReadOnly(true);
                }
            }
        }
    }

    // ROI Mesh init
    if (!f_X0_i.isSet())
    {
        serr<<"f_X0_i is not set"<<sendl;
        BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                f_X0_i.setParent(parent);
                f_X0_i.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader,BaseContext::Local); // perso
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    f_X0_i.setParent(parent);
                    f_X0_i.setReadOnly(true);
                }
            }
        }
    }
    if (!f_edges_i.isSet() || !f_triangles_i.isSet() )
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local); // perso
        if (topology)
        {
            if (!f_edges_i.isSet() && f_computeEdges.getValue())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    f_edges_i.setParent(eparent);
                    f_edges_i.setReadOnly(true);
                }
            }
            if (!f_triangles_i.isSet() && f_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    f_triangles_i.setParent(tparent);
                    f_triangles_i.setReadOnly(true);
                }
            }
        }
    }
    // Bounding Box computation
    Vec6 b=f_box.getValue();
    helper::ReadAccessor< Data<VecCoord > > points_i = f_X0_i;
    if(points_i.size()>0)
    {
        CPos p = DataTypes::getCPos(points_i[0]);
        b[0] = p[0]; b[1] = p[1]; b[2] = p[2];
        b[3] = p[0]; b[4] = p[1]; b[5] = p[2];
        for (unsigned int i=1; i<points_i.size() ; ++i)
        {
            p = DataTypes::getCPos(points_i[i]);
            if (b[0] < p[0]) b[0] = p[0];
            if (b[1] < p[1]) b[1] = p[1];
            if (b[2] < p[2]) b[2] = p[2];
            if (b[3] > p[0]) b[3] = p[0];
            if (b[4] > p[1]) b[4] = p[1];
            if (b[5] > p[2]) b[5] = p[2];
        }
    }
    if (b[0] > b[3]) std::swap(b[0],b[3]);
    if (b[1] > b[4]) std::swap(b[1],b[4]);
    if (b[2] > b[5]) std::swap(b[2],b[5]);
    f_box.setValue(b, true);
    sout << "Bounding Box " << b << sendl;

    // fin perso : init de la mesh template
    first.setValue(1, true); // perso

    addInput(&f_X0);
    addInput(&f_edges);
    addInput(&f_triangles);
    addInput(&f_tetrahedra);

    addInput(&f_X0_i);
    addInput(&f_edges_i);
    addInput(&f_triangles_i);

    addOutput(&f_box);
    addOutput(&f_indices);
    addOutput(&f_edgeIndices);
    addOutput(&f_triangleIndices);
    addOutput(&f_tetrahedronIndices);
    addOutput(&f_pointsInROI);
    addOutput(&f_edgesInROI);
    addOutput(&f_trianglesInROI);
    addOutput(&f_tetrahedraInROI);

    addOutput(&f_pointsOutROI);
    addOutput(&f_edgesOutROI);
    addOutput(&f_trianglesOutROI);
    addOutput(&f_tetrahedraOutROI);
    addOutput(&f_indicesOut);
    addOutput(&f_edgeOutIndices);
    addOutput(&f_triangleOutIndices);
    addOutput(&f_tetrahedronOutIndices);

    setDirtyValue();

    // cerr<<"MeshROI<DataTypes>::init() -> f_X0 = "<<f_X0_i<<endl;
    // cerr<<"MeshROI<DataTypes>::init() -> meshes = "<<meshes<<endl;
    // cerr<<"MeshROI<DataTypes>::init() -> f_indices = "<<f_indices<<endl;
}

template <class DataTypes>
void MeshROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool MeshROI<DataTypes>::CheckSameOrder(const typename DataTypes::CPos& A, const typename DataTypes::CPos& B, const typename DataTypes::CPos& pt, const typename DataTypes::CPos& N)
{
    typename DataTypes::CPos vectorial;
    vectorial[0] = (((B[1] - A[1])*(pt[2] - A[2])) - ((pt[1] - A[1])*(B[2] - A[2])));
    vectorial[1] = (((B[2] - A[2])*(pt[0] - A[0])) - ((pt[2] - A[2])*(B[0] - A[0])));
    vectorial[2] = (((B[0] - A[0])*(pt[1] - A[1])) - ((pt[0] - A[0])*(B[1] - A[1])));
    if( (vectorial[0]*N[0] + vectorial[1]*N[1] + vectorial[2]*N[2]) < 0) return false;
    else return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInMesh(const typename DataTypes::CPos& p)
{
    if(!f_computeTemplateTriangles.getValue()) return true;
// Compute the reference point outside the bounding box
    const Vec6 b = f_box.getValue();
    typename DataTypes::CPos Vec;
    if (sqrt( (b[0]-p[0])*(b[0]-p[0]) + (b[1]-p[1])*(b[1]-p[1]) + (b[2]-p[2])*(b[2]-p[2]) ) < sqrt( (b[3]-p[0])*(b[3]-p[0]) + (b[4]-p[1])*(b[4]-p[1]) + (b[5]-p[2])*(b[5]-p[2]) ) )
    {Vec[0] = (b[0]-100.0f)-p[0] ; Vec[1]= (b[1]-100.0f)-p[1]; Vec[2]= (b[2]-100.0f)-p[2];}
    else
    {Vec[0] = (b[3]+100.0f)-p[0] ; Vec[1]= (b[4]+100.0f)-p[1]; Vec[2]= (b[5]+100.0f)-p[2];}
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles_i = f_triangles_i;
    const VecCoord* x0 = &f_X0_i.getValue();
    int Through=0;
    double d=0.0;
    for (unsigned int i=0; i<triangles_i.size() ; ++i)
    {
        Triangle t = triangles_i[i];
        CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
        CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
        CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
        // Normal N compuation of the ROI mesh triangle
        typename DataTypes::CPos N;
        N[0] = (p1[1]-p0[1])*(p2[2]-p1[2]) - (p1[2]-p0[2])*(p2[1]-p1[1]);
        N[1] = (p1[2]-p0[2])*(p2[0]-p1[0]) - (p1[0]-p0[0])*(p2[2]-p1[2]);
        N[2] = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0]);
        // DotProd computation
        double DotProd = double (N[0]*Vec[0] + N[1]*Vec[1] + N[2]*Vec[2]);
        //std::cout << "Dot Prod " << DotProd << "\t" ;
        if(DotProd !=0)
        {
            // Intersect point with triangle and distance
            d = (N[0]*(p0[0]-p[0])+N[1]*(p0[1]-p[1])+N[2]*(p0[2]-p[2])) / (N[0]*Vec[0]+N[1]*Vec[1]+N[2]*Vec[2]);
            // d negative means that line comes beind the triangle ...
            if(d>=0)
            {
                typename DataTypes::CPos ptIN ;
                ptIN[0] = (Real)(p[0] + d*Vec[0]);
                ptIN[1] = (Real)(p[1] + d*Vec[1]);
                ptIN[2] = (Real)(p[2] + d*Vec[2]);
                if(CheckSameOrder(p0,p1,ptIN,N)) { if(CheckSameOrder(p1,p2,ptIN,N)) { if(CheckSameOrder(p2,p0,ptIN,N)) { Through++; } } }
            }
        }
    }
    if(Through%2!=0)
    {
        // std::cout << "pt ("<<p<<") is INSIDE the MESH!! "<<Through<< std::endl;
        return true;
    }
    // std::cout << "pt ("<<p<<") is outside the MESH!! "<<Through<<" d "<<d<<  std::endl;
    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInMesh(const typename DataTypes::CPos& p, const Vec6& b)
{
    if( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] ) return( isPointInMesh(p) );
    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInMesh(const PointID& pid, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( isPointInMesh(p,b) );
}

template <class DataTypes>
bool MeshROI<DataTypes>::isEdgeInMesh(const Edge& e, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInMesh(c,b);
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTriangleInMesh(const Triangle& t, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInMesh(c,b));
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTetrahedronInMesh(const Tetra &t, const Vec6 &b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInMesh(c,b));
}

template <class DataTypes>
void MeshROI<DataTypes>::update()
{
    if(first.getValue() || p_doUpdate.getValue() )
    {
        first.setValue(false, true);

        cleanDirty();

        // Read accessor for input topology
        helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
        helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
        helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;
//        helper::ReadAccessor< Data<helper::vector<Edge> > > edges_i = f_edges_i;
//        helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles_i = f_triangles_i;

        // Write accessor for topological element indices in MESH
        SetIndex& indices = *f_indices.beginEdit();
        SetIndex& edgeIndices = *f_edgeIndices.beginEdit();
        SetIndex& triangleIndices = *f_triangleIndices.beginEdit();
        SetIndex& tetrahedronIndices = *f_tetrahedronIndices.beginEdit();
        SetIndex& indicesOut = *f_indicesOut.beginEdit();
        SetIndex& edgeOutIndices = *f_edgeOutIndices.beginEdit();
        SetIndex& triangleOutIndices = *f_triangleOutIndices.beginEdit();
        SetIndex& tetrahedronOutIndices = *f_tetrahedronOutIndices.beginEdit();

        // Write accessor for toplogical element in MESH
        helper::WriteAccessor< Data<Vec6> > box = f_box;
        helper::WriteAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        helper::WriteAccessor< Data<helper::vector<Edge> > > edgesInROI = f_edgesInROI;
        helper::WriteAccessor< Data<helper::vector<Triangle> > > trianglesInROI = f_trianglesInROI;
        helper::WriteAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;
        helper::WriteAccessor< Data<VecCoord > > pointsOutROI = f_pointsOutROI;
        helper::WriteAccessor< Data<helper::vector<Edge> > > edgesOutROI = f_edgesOutROI;
        helper::WriteAccessor< Data<helper::vector<Triangle> > > trianglesOutROI = f_trianglesOutROI;
        helper::WriteAccessor< Data<helper::vector<Tetra> > > tetrahedraOutROI = f_tetrahedraOutROI;

        // Clear lists
        indices.clear();
        edgeIndices.clear();
        triangleIndices.clear();
        tetrahedronIndices.clear();

        pointsInROI.clear();
        edgesInROI.clear();
        trianglesInROI.clear();
        tetrahedraInROI.clear();
        indicesOut.clear();
        edgeOutIndices.clear();
        triangleOutIndices.clear();
        tetrahedronOutIndices.clear();

        pointsOutROI.clear();
        edgesOutROI.clear();
        trianglesOutROI.clear();
        tetrahedraOutROI.clear();


        const VecCoord* x0 = &f_X0.getValue();
        const Vec6 b=f_box.getValue();
        //Points
        for( unsigned i=0; i<x0->size(); ++i )
        {
            if (isPointInMesh(i, b))
            {
                indices.push_back(i);
                pointsInROI.push_back((*x0)[i]);
//         sout<<"\nMeshROI<DataTypes>::update, add index "<< i << sendl;
            }
            else
            {
                indicesOut.push_back(i);
                pointsOutROI.push_back((*x0)[i]);
//         sout<<"\nMeshROI<DataTypes>::update, add index "<< i << sendl;
            }
        }

        //Edges
        if (f_computeEdges.getValue())
        {
            for(unsigned int i=0 ; i<edges.size() ; i++)
            {
                Edge e = edges[i];
                if (isEdgeInMesh(e, b))
                {
                    edgeIndices.push_back(i);
                    edgesInROI.push_back(e);
                }
                else
                {
                    edgeOutIndices.push_back(i);
                    edgesOutROI.push_back(e);
                }
            }
        }

        //Triangles
        if (f_computeTriangles.getValue())
        {
            for(unsigned int i=0 ; i<triangles.size() ; i++)
            {
                Triangle t = triangles[i];
                if (isTriangleInMesh(t, b))
                {
                    triangleIndices.push_back(i);
                    trianglesInROI.push_back(t);
                }
                else
                {
                    triangleOutIndices.push_back(i);
                    trianglesOutROI.push_back(t);
                }
            }
        }

        //Tetrahedra
        if (f_computeTetrahedra.getValue())
        {
            for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
            {
                Tetra t = tetrahedra[i];
                if (isTetrahedronInMesh(t, b))
                {
                    tetrahedronIndices.push_back(i);
                    tetrahedraInROI.push_back(t);
                }
                else
                {
                    tetrahedronOutIndices.push_back(i);
                    tetrahedraOutROI.push_back(t);
                }
            }
        }
        f_indices.endEdit();
        f_edgeIndices.endEdit();
        f_triangleIndices.endEdit();
        f_tetrahedronIndices.endEdit();

        f_indicesOut.endEdit();
        f_edgeOutIndices.endEdit();
        f_triangleOutIndices.endEdit();
        f_tetrahedronOutIndices.endEdit();

    }
}


template <class DataTypes>
void MeshROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels() && !this->_drawSize.getValue())
        return;

    const VecCoord* x0 = &f_X0.getValue();

    glColor3f(1.0f, 0.4f, 0.4f);

    // draw the ROI mesh
    if( p_drawMesh.getValue())
    {
        glColor3f(0.4f, 0.4f, 1.0f);
        const VecCoord* x0_i = &f_X0_i.getValue();
        ///draw ROI points
        if(p_drawPoints.getValue())
        {
            if (_drawSize.getValue())
                glPointSize((GLfloat)_drawSize.getValue());
            glDisable(GL_LIGHTING);
            glBegin(GL_POINTS);
            glPointSize(5.0);
            helper::ReadAccessor< Data<VecCoord > > points_i = f_X0_i;
            for (unsigned int i=0; i<points_i.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(points_i[i]);
                helper::gl::glVertexT(p);
            }
            glEnd();
            glPointSize(1);
        }
        /// draw ROI edges
        if(p_drawEdges.getValue())
        {
            glDisable(GL_LIGHTING);
            glLineWidth((GLfloat)_drawSize.getValue());
            glBegin(GL_LINES);
            helper::ReadAccessor< Data<helper::vector<Edge> > > edges_i = f_edges_i;
            for (unsigned int i=0; i<edges_i.size() ; ++i)
            {
                Edge e = edges_i[i];
                for (unsigned int j=0 ; j<2 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0_i)[e[j]]);
                    helper::gl::glVertexT(p);
                }
            }
            glEnd();
            glPointSize(1);
        }
        // draw ROI triangles
        if(p_drawTriangles.getValue())
        {
            glDisable(GL_LIGHTING);
            glBegin(GL_TRIANGLES);
            helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles_i = f_triangles_i;
            for (unsigned int i=0; i<triangles_i.size() ; ++i)
            {
                Triangle t = triangles_i[i];
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0_i)[t[j]]);
                    helper::gl::glVertexT(p);
                }
            }
            glEnd();
        }
        glColor3f(1.0f, 0.4f, 0.4f);
    }
    // draw the bounding box
    if( p_drawBox.getValue())
    {
        glDisable(GL_LIGHTING);
        if (_drawSize.getValue())
            glLineWidth((GLfloat)_drawSize.getValue());
        glBegin(GL_LINES);
        const Vec6& b=f_box.getValue();
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
        if(p_drawOut.getValue())
        {
            helper::ReadAccessor< Data<VecCoord > > pointsROI = f_pointsOutROI;
            for (unsigned int i=0; i<pointsROI.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(pointsROI[i]);
                helper::gl::glVertexT(p);
            }
        }
        else
        {
            helper::ReadAccessor< Data<VecCoord > > pointsROI = f_pointsInROI;
            for (unsigned int i=0; i<pointsROI.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(pointsROI[i]);
                helper::gl::glVertexT(p);
            }
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
        if(p_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Edge> > > edgesROI = f_edgesOutROI;
            for (unsigned int i=0; i<edgesROI.size() ; ++i)
            {
                Edge e = edgesROI[i];
                for (unsigned int j=0 ; j<2 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0)[e[j]]);
                    helper::gl::glVertexT(p);
                }
            }
        }
        else
        {
            helper::ReadAccessor< Data<helper::vector<Edge> > > edgesROI = f_edgesInROI;
            for (unsigned int i=0; i<edgesROI.size() ; ++i)
            {
                Edge e = edgesROI[i];
                for (unsigned int j=0 ; j<2 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0)[e[j]]);
                    helper::gl::glVertexT(p);
                }
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
        if(p_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesROI = f_trianglesOutROI;
            for (unsigned int i=0; i<trianglesROI.size() ; ++i)
            {
                Triangle t = trianglesROI[i];
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0)[t[j]]);
                    helper::gl::glVertexT(p);
                }
            }
        }
        else
        {
            helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesROI = f_trianglesInROI;
            for (unsigned int i=0; i<trianglesROI.size() ; ++i)
            {
                Triangle t = trianglesROI[i];
                for (unsigned int j=0 ; j<3 ; j++)
                {
                    CPos p = DataTypes::getCPos((*x0)[t[j]]);
                    helper::gl::glVertexT(p);
                }
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
        if(p_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraROI = f_tetrahedraOutROI;
            for (unsigned int i=0; i<tetrahedraROI.size() ; ++i)
            {
                Tetra t = tetrahedraROI[i];
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
        }
        else
        {
            helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraROI = f_tetrahedraInROI;
            for (unsigned int i=0; i<tetrahedraROI.size() ; ++i)
            {
                Tetra t = tetrahedraROI[i];
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
        }
        glEnd();
        glLineWidth(1);
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
