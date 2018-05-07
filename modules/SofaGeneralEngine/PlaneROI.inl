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
#ifndef SOFA_COMPONENT_ENGINE_PLANEROI_INL
#define SOFA_COMPONENT_ENGINE_PLANEROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/PlaneROI.h>
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
PlaneROI<DataTypes>::PlaneROI()
    : planes( initData(&planes, "plane", "Plane defined by 3 points and a depth distance") )
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
    planes.beginEdit()->push_back(Vec10(sofa::defaulttype::Vec<9,Real>(0,0,0,0,0,0,0,0,0),0));
    planes.endEdit();

    addAlias(&f_X0,"rest_position");
    addAlias(&p_drawBoxes,"isVisible");

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void PlaneROI<DataTypes>::init()
{
    using sofa::core::objectmodel::BaseData;

    if (!f_X0.isSet())
    {
        sofa::core::behavior::MechanicalState<DataTypes>* mstate;
        this->getContext()->get(mstate);
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
            this->getContext()->get(loader);
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
    if (!f_edges.isSet() || !f_triangles.isSet())
    {
        sofa::core::topology::BaseMeshTopology* topology;
        this->getContext()->get(topology);
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
}

template <class DataTypes>
void PlaneROI<DataTypes>::reinit()
{
    update();
}


template <class DataTypes>
void PlaneROI<DataTypes>::computePlane(unsigned int planeIndex)
{
    const helper::vector<Vec10>& vp=planes.getValue();
    const Vec10& p=vp[planeIndex];

    p0 = Vec3(p[0], p[1], p[2]);
    p1 = Vec3(p[3], p[4], p[5]);
    p2 = Vec3(p[6], p[7], p[8]);
    depth = p[9];

    vdepth = (p1-p0).cross(p2-p0);
    vdepth.normalize();

    p3 = p0 + (p2-p1);
    p4 = p0 + vdepth * (depth/2);
    p6 = p2 + vdepth * (depth/2);

    plane0 = (p1-p0).cross(p4-p0);
    plane0.normalize();

    plane1 = (p2-p3).cross(p6-p3);
    plane1.normalize();

    plane2 = (p3-p0).cross(p4-p0);
    plane2.normalize();

    plane3 = (p2-p1).cross(p6-p2);
    plane3.normalize();

    width = fabs(dot((p2-p0),plane0));
    length = fabs(dot((p2-p0),plane2));

}



template <class DataTypes>
bool PlaneROI<DataTypes>::isPointInPlane(const typename DataTypes::CPos& p)
{
    Vec3 pv0 = (p-p0);
    Vec3 pv1 = (p-p2);

    if( fabs(dot(pv0, plane0)) <= width && fabs(dot(pv1, plane1)) <= width )
    {
        if ( fabs(dot(pv0, plane2)) <= length && fabs(dot(pv1, plane3)) <= length )
        {
            if ( !(fabs(dot(pv0, vdepth)) <= fabs(depth/2)) )
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    return true;
}


template <class DataTypes>
bool PlaneROI<DataTypes>::isPointInPlane(const PointID& pid)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( isPointInPlane(p) );
}


template <class DataTypes>
bool PlaneROI<DataTypes>::isEdgeInPlane(const Edge& e)
{
    const VecCoord* x0 = &f_X0.getValue();
    for (unsigned int i=0; i<2; ++i)
    {
        CPos p =  DataTypes::getCPos((*x0)[e[i]]);
        if (!isPointInPlane(p))
            return false;
    }
    return true;
}


template <class DataTypes>
bool PlaneROI<DataTypes>::isTriangleInPlane(const Triangle& t)
{
    const VecCoord* x0 = &f_X0.getValue();
    for (unsigned int i=0; i<3; ++i)
    {
        CPos p =  DataTypes::getCPos((*x0)[t[i]]);
        if (!isPointInPlane(p))
            return false;
    }
    return true;
}


template <class DataTypes>
bool PlaneROI<DataTypes>::isTetrahedronInPlane(const Tetra& t)
{
    const VecCoord* x0 = &f_X0.getValue();
    for (unsigned int i=0; i<4; ++i)
    {
        CPos p =  DataTypes::getCPos((*x0)[t[i]]);
        if (!isPointInPlane(p))
            return false;
    }
    return true;
}



template <class DataTypes>
void PlaneROI<DataTypes>::update()
{
    const helper::vector<Vec10>& vp=planes.getValue();
    if (vp.empty())
        return;

    // Read accessor for input topology
    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;


    const VecCoord* x0 = &f_X0.getValue();

    cleanDirty();

    // Write accessor for topological element indices in SPHERE
    SetIndex& indices = *(f_indices.beginWriteOnly());
    SetIndex& edgeIndices = *(f_edgeIndices.beginWriteOnly());
    SetIndex& triangleIndices = *(f_triangleIndices.beginWriteOnly());
    SetIndex& tetrahedronIndices = *f_tetrahedronIndices.beginWriteOnly();

    // Write accessor for toplogical element in SPHERE
    helper::WriteOnlyAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Edge> > > edgesInROI = f_edgesInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Triangle> > > trianglesInROI = f_trianglesInROI;
    helper::WriteOnlyAccessor< Data<helper::vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;


    // Clear lists
    indices.clear();
    edgeIndices.clear();
    triangleIndices.clear();
    tetrahedronIndices.clear();

    pointsInROI.clear();
    edgesInROI.clear();
    trianglesInROI.clear();
    tetrahedraInROI.clear();


    //Points
    for( unsigned i=0; i<x0->size(); ++i )
    {
        for (unsigned int j=0; j<vp.size(); ++j)
        {
            this->computePlane(j);
            if (isPointInPlane(i))
            {
                indices.push_back(i);
                pointsInROI.push_back((*x0)[i]);
                break;
            }
        }
    }

    //Edges
    if (f_computeEdges.getValue())
    {
        for(unsigned int i=0 ; i<edges.size() ; i++)
        {
            Edge edge = edges[i];
            for (unsigned int j=0; j<vp.size(); ++j)
            {
                this->computePlane(j);
                if (isEdgeInPlane(edge))
                {
                    edgeIndices.push_back(i);
                    edgesInROI.push_back(edge);
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
            Triangle tri = triangles[i];
            for (unsigned int j=0; j<vp.size(); ++j)
            {
                this->computePlane(j);
                if (isTriangleInPlane(tri))
                {
                    triangleIndices.push_back(i);
                    trianglesInROI.push_back(tri);
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
            for (unsigned int j=0; j<vp.size(); ++j)
            {
                this->computePlane(j);
                if (isTetrahedronInPlane(t))
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
void PlaneROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const VecCoord* x0 = &f_X0.getValue();
    glColor3f(0.0, 1.0, 1.0);

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        ///draw the boxes
        glBegin(GL_LINES);
        const helper::vector<Vec10>& vp=planes.getValue();
        for (unsigned int pi=0; pi<vp.size(); ++pi)
        {
            const Vec10& p=vp[pi];
            Vec3 p0 = Vec3(p[0], p[1], p[2]);
            Vec3 p1 = Vec3(p[3], p[4], p[5]);
            Vec3 p2 = Vec3(p[6], p[7], p[8]);

            Vec3 p3 = p0 + (p2 - p1);

            Vec3 n = (p1-p0).cross(p2-p0);
            n.normalize();
            const Real depth = p[9];

            Vec3 p4 = p0 + n * (depth/2);
            p0 = p0 + (-n) * (depth/2);
            Vec3 p5 = p1 + n * (depth/2);
            p1 = p1 + (-n) * (depth/2);
            Vec3 p6 = p2 + n * (depth/2);
            p2 = p2 + (-n) * (depth/2);
            Vec3 p7 = p3 + n * (depth/2);
            p3 = p3 + (-n) * (depth/2);

            glVertex3d(p0.x(), p0.y(), p0.z());
            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p0.x(), p0.y(), p0.z());

            glVertex3d(p4.x(), p4.y(), p4.z());
            glVertex3d(p5.x(), p5.y(), p5.z());
            glVertex3d(p5.x(), p5.y(), p5.z());
            glVertex3d(p6.x(), p6.y(), p6.z());
            glVertex3d(p6.x(), p6.y(), p6.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
            glVertex3d(p4.x(), p4.y(), p4.z());

            glVertex3d(p0.x(), p0.y(), p0.z());
            glVertex3d(p4.x(), p4.y(), p4.z());

            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p5.x(), p5.y(), p5.z());

            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p6.x(), p6.y(), p6.z());

            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
        }
        glEnd();
    }

    ///draw points in ROI
    if( p_drawPoints.getValue())
    {
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
    }

    ///draw triangles in ROI
    if( p_drawTriangles.getValue())
    {
        glDisable(GL_LIGHTING);
        glLineWidth((GLfloat)_drawSize.getValue());
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
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
