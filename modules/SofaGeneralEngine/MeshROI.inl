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
#ifndef SOFA_COMPONENT_ENGINE_MESHROI_INL
#define SOFA_COMPONENT_ENGINE_MESHROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/MeshROI.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

namespace engine
{

using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::BaseContext;
using sofa::core::topology::BaseMeshTopology;
using sofa::core::behavior::BaseMechanicalState;
using core::loader::MeshLoader;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteOnlyAccessor;
using sofa::helper::vector;
using core::visual::VisualParams;

template <class DataTypes>
MeshROI<DataTypes>::MeshROI()
    : d_X0( initData (&d_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , d_edges(initData (&d_edges, "edges", "Edge Topology") )
    , d_triangles(initData (&d_triangles, "triangles", "Triangle Topology") )
    , d_tetrahedra(initData (&d_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , d_X0_i( initData (&d_X0_i, "ROIposition", "ROI position coordinates of the degrees of freedom") )
    , d_edges_i(initData (&d_edges_i, "ROIedges", "ROI Edge Topology") )
    , d_triangles_i(initData (&d_triangles_i, "ROItriangles", "ROI Triangle Topology") )
    , d_computeEdges( initData(&d_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI.") )
    , d_computeTriangles( initData(&d_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI.") )
    , d_computeTetrahedra( initData(&d_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI.") )
    , d_computeTemplateTriangles( initData(&d_computeTemplateTriangles,true,"computeMeshROI","Compute with the mesh (not only bounding box)") )
    , d_box( initData(&d_box, "box", "Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , d_indices( initData(&d_indices,"indices","Indices of the points contained in the ROI") )
    , d_edgeIndices( initData(&d_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , d_triangleIndices( initData(&d_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , d_tetrahedronIndices( initData(&d_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , d_pointsInROI( initData(&d_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , d_edgesInROI( initData(&d_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , d_trianglesInROI( initData(&d_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , d_tetrahedraInROI( initData(&d_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , d_pointsOutROI( initData(&d_pointsOutROI,"pointsOutROI","Points not contained in the ROI") )
    , d_edgesOutROI( initData(&d_edgesOutROI,"edgesOutROI","Edges not contained in the ROI") )
    , d_trianglesOutROI( initData(&d_trianglesOutROI,"trianglesOutROI","Triangles not contained in the ROI") )
    , d_tetrahedraOutROI( initData(&d_tetrahedraOutROI,"tetrahedraOutROI","Tetrahedra not contained in the ROI") )
    , d_indicesOut( initData(&d_indicesOut,"indicesOut","Indices of the points not contained in the ROI") )
    , d_edgeOutIndices( initData(&d_edgeOutIndices,"edgeOutIndices","Indices of the edges not contained in the ROI") )
    , d_triangleOutIndices( initData(&d_triangleOutIndices,"triangleOutIndices","Indices of the triangles not contained in the ROI") )
    , d_tetrahedronOutIndices( initData(&d_tetrahedronOutIndices,"tetrahedronOutIndices","Indices of the tetrahedra not contained in the ROI") )
    , d_drawOut( initData(&d_drawOut,false,"drawOut","Draw the data not contained in the ROI") )
    , d_drawMesh( initData(&d_drawMesh,false,"drawMesh","Draw Mesh used for the ROI") )
    , d_drawBox( initData(&d_drawBox,false,"drawBox","Draw the Bounding box around the mesh used for the ROI") )
    , d_drawPoints( initData(&d_drawPoints,false,"drawPoints","Draw Points") )
    , d_drawEdges( initData(&d_drawEdges,false,"drawEdges","Draw Edges") )
    , d_drawTriangles( initData(&d_drawTriangles,false,"drawTriangles","Draw Triangles") )
    , d_drawTetrahedra( initData(&d_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra") )
    , d_drawSize( initData(&d_drawSize,0.0,"drawSize","rendering size for mesh and topological elements") )
    , d_doUpdate( initData(&d_doUpdate,false,"doUpdate","Update the computation (not only at the init)") )
{
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();
}

template <class DataTypes>
void MeshROI<DataTypes>::init()
{
    addInput(&d_X0);
    addInput(&d_edges);
    addInput(&d_triangles);
    addInput(&d_tetrahedra);

    addInput(&d_X0_i);
    addInput(&d_edges_i);
    addInput(&d_triangles_i);

    addOutput(&d_box);
    addOutput(&d_indices);
    addOutput(&d_edgeIndices);
    addOutput(&d_triangleIndices);
    addOutput(&d_tetrahedronIndices);
    addOutput(&d_pointsInROI);
    addOutput(&d_edgesInROI);
    addOutput(&d_trianglesInROI);
    addOutput(&d_tetrahedraInROI);

    addOutput(&d_pointsOutROI);
    addOutput(&d_edgesOutROI);
    addOutput(&d_trianglesOutROI);
    addOutput(&d_tetrahedraOutROI);
    addOutput(&d_indicesOut);
    addOutput(&d_edgeOutIndices);
    addOutput(&d_triangleOutIndices);
    addOutput(&d_tetrahedronOutIndices);

    setDirtyValue();

    checkInputData();
    computeBoundingBox();
    compute();
}


template <class DataTypes>
void MeshROI<DataTypes>::checkInputData()
{
    if (!d_X0.isSet())
    {
        msg_warning(this) << "Data 'position' is not set. Get rest position of local mechanical state "
                          << "or mesh loader (if no mechanical)";

        BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                d_X0.setParent(parent);
                d_X0.setReadOnly(true);
            }
        }
        else
        {
            MeshLoader* loader = NULL;
            this->getContext()->get(loader,BaseContext::Local); // perso
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    d_X0.setParent(parent);
                    d_X0.setReadOnly(true);
                }
            }
        }
    }

    if (!d_edges.isSet() || !d_triangles.isSet() || !d_tetrahedra.isSet())
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local); // perso
        if (topology)
        {
            if (!d_edges.isSet() && d_computeEdges.getValue())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    d_edges.setParent(eparent);
                    d_edges.setReadOnly(true);
                }
            }
            if (!d_triangles.isSet() && d_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    d_triangles.setParent(tparent);
                    d_triangles.setReadOnly(true);
                }
            }
            if (!d_tetrahedra.isSet() && d_computeTetrahedra.getValue())
            {
                BaseData* tparent = topology->findData("tetrahedra");
                if (tparent)
                {
                    d_tetrahedra.setParent(tparent);
                    d_tetrahedra.setReadOnly(true);
                }
            }
        }
    }

    // ROI Mesh init
    if (!d_X0_i.isSet())
    {
        msg_warning(this) << "Data 'ROIposition' is not set. Get rest position of local mechanical state "
                          << "or mesh loader (if no mechanical)";

        BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                d_X0_i.setParent(parent);
                d_X0_i.setReadOnly(true);
            }
        }
        else
        {
            MeshLoader* loader = NULL;
            this->getContext()->get(loader,BaseContext::Local); // perso
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    d_X0_i.setParent(parent);
                    d_X0_i.setReadOnly(true);
                }
            }
        }
    }

    if (!d_edges_i.isSet() || !d_triangles_i.isSet() )
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local); // perso
        if (topology)
        {
            if (!d_edges_i.isSet() && d_computeEdges.getValue())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    d_edges_i.setParent(eparent);
                    d_edges_i.setReadOnly(true);
                }
            }
            if (!d_triangles_i.isSet() && d_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    d_triangles_i.setParent(tparent);
                    d_triangles_i.setReadOnly(true);
                }
            }
        }
    }
}


template <class DataTypes>
void MeshROI<DataTypes>::computeBoundingBox()
{
    // Bounding Box computation
    Vec6 b=d_box.getValue();
    ReadAccessor<Data<VecCoord>> points_i = d_X0_i;
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
    d_box.setValue(b);

    msg_info(this) << "Bounding Box " << b;
}


template <class DataTypes>
void MeshROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool MeshROI<DataTypes>::checkSameOrder(const typename DataTypes::CPos& A, const typename DataTypes::CPos& B, const typename DataTypes::CPos& pt, const typename DataTypes::CPos& N)
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
    if(!d_computeTemplateTriangles.getValue()) return true;

    if(isPointInBoundingBox(p))
    {
        // Compute the reference point outside the bounding box
        const Vec6 b = d_box.getValue();
        typename DataTypes::CPos Vec;
        if (( (b[0]-p[0])*(b[0]-p[0]) + (b[1]-p[1])*(b[1]-p[1]) + (b[2]-p[2])*(b[2]-p[2]) ) < ( (b[3]-p[0])*(b[3]-p[0]) + (b[4]-p[1])*(b[4]-p[1]) + (b[5]-p[2])*(b[5]-p[2]) ) )
        {
            Vec[0]= (b[0]-100.0f)-p[0] ;
            Vec[1]= (b[1]-100.0f)-p[1];
            Vec[2]= (b[2]-100.0f)-p[2];
        }
        else
        {
            Vec[0]= (b[3]+100.0f)-p[0] ;
            Vec[1]= (b[4]+100.0f)-p[1];
            Vec[2]= (b[5]+100.0f)-p[2];
        }

        ReadAccessor< Data<vector<Triangle> > > triangles_i = d_triangles_i;
        const VecCoord* x0 = &d_X0_i.getValue();
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
                    if(checkSameOrder(p0,p1,ptIN,N)) { if(checkSameOrder(p1,p2,ptIN,N)) { if(checkSameOrder(p2,p0,ptIN,N)) { Through++; } } }
                }
            }
        }
        if(Through%2!=0)
            return true;
    }

    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInIndices(const unsigned int &pointId)
{
    ReadAccessor<Data<SetIndex>> indices = d_indices;

    for (unsigned int i=0; i<indices.size(); i++)
        if(indices[i]==pointId)
            return true;

    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInBoundingBox(const typename DataTypes::CPos& p)
{
    const Vec6 b = d_box.getValue();
    if( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] )
        return true;
    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isEdgeInMesh(const Edge& e)
{
    for (int i=0; i<2; i++)
        if(!isPointInIndices(e[i]))
        {
            const VecCoord* x0 = &d_X0.getValue();
            CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
            CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
            CPos c = (p1+p0)*0.5;

            return (isPointInMesh(c));
        }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTriangleInMesh(const Triangle& t)
{
    for (int i=0; i<3; i++)
        if(!isPointInIndices(t[i]))
        {
            const VecCoord* x0 = &d_X0.getValue();
            CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
            CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
            CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
            CPos c = (p2+p1+p0)/3.0;

            return (isPointInMesh(c));
        }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTetrahedronInMesh(const Tetra &t)
{
    for (int i=0; i<4; i++)
        if(!isPointInIndices(t[i]))
        {
            const VecCoord* x0 = &d_X0.getValue();
            CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
            CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
            CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
            CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
            CPos c = (p3+p2+p1+p0)/4.0;

            return (isPointInMesh(c));
        }

    return true;
}

template <class DataTypes>
void MeshROI<DataTypes>::compute()
{
    // Read accessor for input topology
    ReadAccessor< Data<vector<Edge> > > edges = d_edges;
    ReadAccessor< Data<vector<Triangle> > > triangles = d_triangles;
    ReadAccessor< Data<vector<Tetra> > > tetrahedra = d_tetrahedra;

    updateAllInputsIfDirty(); // the easy way to make sure every inputs are up-to-date

    cleanDirty();

    // Write accessor for topological element indices in MESH
    SetIndex& indices = *d_indices.beginWriteOnly();
    SetIndex& edgeIndices = *d_edgeIndices.beginWriteOnly();
    SetIndex& triangleIndices = *d_triangleIndices.beginWriteOnly();
    SetIndex& tetrahedronIndices = *d_tetrahedronIndices.beginWriteOnly();
    SetIndex& indicesOut = *d_indicesOut.beginWriteOnly();
    SetIndex& edgeOutIndices = *d_edgeOutIndices.beginWriteOnly();
    SetIndex& triangleOutIndices = *d_triangleOutIndices.beginWriteOnly();
    SetIndex& tetrahedronOutIndices = *d_tetrahedronOutIndices.beginWriteOnly();

    // Write accessor for toplogical element in MESH
    WriteOnlyAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
    WriteOnlyAccessor< Data<vector<Edge> > > edgesInROI = d_edgesInROI;
    WriteOnlyAccessor< Data<vector<Triangle> > > trianglesInROI = d_trianglesInROI;
    WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
    WriteOnlyAccessor< Data<VecCoord > > pointsOutROI = d_pointsOutROI;
    WriteOnlyAccessor< Data<vector<Edge> > > edgesOutROI = d_edgesOutROI;
    WriteOnlyAccessor< Data<vector<Triangle> > > trianglesOutROI = d_trianglesOutROI;
    WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraOutROI = d_tetrahedraOutROI;

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


    const VecCoord* x0 = &d_X0.getValue();
    //Points
    for( unsigned i=0; i<x0->size(); ++i )
    {
        CPos p =  DataTypes::getCPos((*x0)[i]);
        if (isPointInMesh(p))
        {
            indices.push_back(i);
            pointsInROI.push_back((*x0)[i]);
        }
        else
        {
            indicesOut.push_back(i);
            pointsOutROI.push_back((*x0)[i]);
        }
    }

    //Edges
    if (d_computeEdges.getValue())
    {
        for(unsigned int i=0 ; i<edges.size() ; i++)
        {
            Edge e = edges[i];
            if (isEdgeInMesh(e))
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
    if (d_computeTriangles.getValue())
    {
        for(unsigned int i=0 ; i<triangles.size() ; i++)
        {
            Triangle t = triangles[i];
            if (isTriangleInMesh(t))
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
    if (d_computeTetrahedra.getValue())
    {
        for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
        {
            Tetra t = tetrahedra[i];
            if (isTetrahedronInMesh(t))
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
    d_indices.endEdit();
    d_edgeIndices.endEdit();
    d_triangleIndices.endEdit();
    d_tetrahedronIndices.endEdit();

    d_indicesOut.endEdit();
    d_edgeOutIndices.endEdit();
    d_triangleOutIndices.endEdit();
    d_tetrahedronOutIndices.endEdit();
}


template <class DataTypes>
void MeshROI<DataTypes>::update()
{
    if(d_doUpdate.getValue())
        compute();
}


template <class DataTypes>
void MeshROI<DataTypes>::draw(const VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels() && !this->d_drawSize.getValue())
        return;

    const VecCoord* x0 = &d_X0.getValue();

    glColor3f(1.0f, 0.4f, 0.4f);

    // draw the ROI mesh
    if( d_drawMesh.getValue())
    {
        glColor3f(0.4f, 0.4f, 1.0f);
        const VecCoord* x0_i = &d_X0_i.getValue();
        ///draw ROI points
        if(d_drawPoints.getValue())
        {
            if (d_drawSize.getValue())
                glPointSize((GLfloat)d_drawSize.getValue());
            glDisable(GL_LIGHTING);
            glBegin(GL_POINTS);
            glPointSize(5.0);
            helper::ReadAccessor< Data<VecCoord > > points_i = d_X0_i;
            for (unsigned int i=0; i<points_i.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(points_i[i]);
                helper::gl::glVertexT(p);
            }
            glEnd();
            glPointSize(1);
        }
        // draw ROI edges
        if(d_drawEdges.getValue())
        {
            glDisable(GL_LIGHTING);
            glLineWidth((GLfloat)d_drawSize.getValue());
            glBegin(GL_LINES);
            helper::ReadAccessor< Data<helper::vector<Edge> > > edges_i = d_edges_i;
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
        if(d_drawTriangles.getValue())
        {
            glDisable(GL_LIGHTING);
            glBegin(GL_TRIANGLES);
            helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles_i = d_triangles_i;
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
    if( d_drawBox.getValue())
    {
        glDisable(GL_LIGHTING);
        if (d_drawSize.getValue())
            glLineWidth((GLfloat)d_drawSize.getValue());
        glBegin(GL_LINES);
        const Vec6& b=d_box.getValue();
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
    // draw points in ROI
    if( d_drawPoints.getValue())
    {
        if (d_drawSize.getValue())
            glPointSize((GLfloat)d_drawSize.getValue());
        glDisable(GL_LIGHTING);
        glBegin(GL_POINTS);
        glPointSize(5.0);
        if(d_drawOut.getValue())
        {
            helper::ReadAccessor< Data<VecCoord > > pointsROI = d_pointsOutROI;
            for (unsigned int i=0; i<pointsROI.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(pointsROI[i]);
                helper::gl::glVertexT(p);
            }
        }
        else
        {
            helper::ReadAccessor< Data<VecCoord > > pointsROI = d_pointsInROI;
            for (unsigned int i=0; i<pointsROI.size() ; ++i)
            {
                CPos p = DataTypes::getCPos(pointsROI[i]);
                helper::gl::glVertexT(p);
            }
        }
        glEnd();
        glPointSize(1);
    }
    // draw edges in ROI
    if( d_drawEdges.getValue())
    {
        glDisable(GL_LIGHTING);
        glLineWidth((GLfloat)d_drawSize.getValue());
        glBegin(GL_LINES);
        if(d_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Edge> > > edgesROI = d_edgesOutROI;
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
            helper::ReadAccessor< Data<helper::vector<Edge> > > edgesROI = d_edgesInROI;
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
    // draw triangles in ROI
    if( d_drawTriangles.getValue())
    {
        glDisable(GL_LIGHTING);
        glBegin(GL_TRIANGLES);
        if(d_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesROI = d_trianglesOutROI;
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
            helper::ReadAccessor< Data<helper::vector<Triangle> > > trianglesROI = d_trianglesInROI;
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
    // draw tetrahedra in ROI
    if( d_drawTetrahedra.getValue())
    {
        glDisable(GL_LIGHTING);
        glLineWidth((GLfloat)d_drawSize.getValue());
        glBegin(GL_LINES);
        if(d_drawOut.getValue())
        {
            helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraROI = d_tetrahedraOutROI;
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
            helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedraROI = d_tetrahedraInROI;
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
