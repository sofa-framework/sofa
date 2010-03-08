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
#ifndef SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_INL
#define SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/PrimitivesInSphereROI.h>
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

template <class DataTypes>
PrimitivesInSphereROI<DataTypes>::PrimitivesInSphereROI()
    : isVisible( initData (&isVisible, bool (true), "isVisible", "is Visible ?") )
    , centers( initData(&centers, "centers", "Center(s) of the sphere(s)") )
    , radii( initData(&radii, "radii", "Radius(i) of the sphere(s)") )
    , direction( initData(&direction, "direction", "Edge direction(if edgeAngle > 0)") )
    , normal( initData(&normal, "normal", "Normal direction of the triangles (if triAngle > 0)") )
    , edgeAngle( initData(&edgeAngle, (Real)0, "edgeAngle", "Max angle between the direction of the selected edges and the specified direction") )
    , triAngle( initData(&triAngle, (Real)0, "triAngle", "Max angle between the normal of the selected triangle and the specified normal direction") )
    , f_X0( initData (&f_X0, "rest_position", "Rest position coordinates of the degrees of freedom") )
    , f_edges(initData(&f_edges, "edges", "List of edge indices"))
    , f_triangles(initData(&f_triangles, "triangles", "List of triangle indices"))
    , f_pointIndices( initData(&f_pointIndices,"pointIndices","Indices of the points contained in the ROI") )
    , f_edgeIndices( initData(&f_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , f_triangleIndices( initData(&f_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering") )
{
    f_triangleIndices.beginEdit()->push_back(0);
    f_triangleIndices.endEdit();
}

template <class DataTypes>
void PrimitivesInSphereROI<DataTypes>::init()
{
    if (f_X0.getValue().empty())
    {
        MechanicalState<DataTypes>* mstate;
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
    }
    if (f_edges.getValue().empty())
    {
        BaseMeshTopology* topology = dynamic_cast<BaseMeshTopology*>(getContext()->getTopology());
        if (topology != NULL)
        {
            pointSize = topology->getNbPoints();
            BaseData* parent = topology->findField("edges");
            if (parent != NULL)
            {
                f_edges.setParent(parent);
                f_edges.setReadOnly(true);
            }
            else
            {
                sout << "ERROR: Topology " << topology->getName() << " does not contain edges" << sendl;
            }
        }
        else
        {
            sout << "ERROR: Topology not found. Edges in sphere can not be computed" << sendl;
        }
    }
    if (f_triangles.getValue().empty())
    {
        BaseMeshTopology* topology = dynamic_cast<BaseMeshTopology*>(getContext()->getTopology());
        if (topology != NULL)
        {
            BaseData* parent = topology->findField("triangles");
            if (parent != NULL)
            {
                f_triangles.setParent(parent);
                f_triangles.setReadOnly(true);
            }
            else
            {
                sout << "ERROR: Topology " << topology->getName() << " does not contain triangles" << sendl;
            }
        }
        else
        {
            sout << "ERROR: Topology not found. Triangles in sphere can not be computed" << sendl;
        }
    }

    addInput(&f_X0);
    addInput(&f_edges);
    addInput(&f_triangles);
    addInput(&centers);
    addInput(&radii);
    addInput(&direction);
    addInput(&normal);
    addInput(&edgeAngle);
    addInput(&triAngle);

    addOutput(&f_pointIndices);
    addOutput(&f_edgeIndices);
    addOutput(&f_triangleIndices);

    setDirtyValue();
}

template <class DataTypes>
void PrimitivesInSphereROI<DataTypes>::reinit()
{
    update();
}


template <class DataTypes>
bool PrimitivesInSphereROI<DataTypes>::containsPoint(const Vec3& c, const Real& r, const Coord& p)
{
    if((p-c).norm() > r)
        return false;
    else
        return true;
}

template <class DataTypes>
bool PrimitivesInSphereROI<DataTypes>::containsEdge(const Vec3& c, const Real& r, const BaseMeshTopology::Edge& edge)
{
    for (unsigned int i=0; i<2; ++i)
    {
        Coord p = (*x0)[edge[i]];

        if((p-c).norm() > r)
            return false;
    }
    return true;
}

template <class DataTypes>
bool PrimitivesInSphereROI<DataTypes>::containsTriangle(const Vec3& c, const Real& r, const BaseMeshTopology::Triangle& triangle)
{
    for (unsigned int i=0; i<3; ++i)
    {
        Coord p = (*x0)[triangle[i]];

        if((p-c).norm() > r)
            return false;
    }
    return true;
}

template <class DataTypes>
void PrimitivesInSphereROI<DataTypes>::update()
{
    cleanDirty();

    const helper::vector<Vec3>& c = (centers.getValue());
    const helper::vector<Real>& r = (radii.getValue());
    Real eAngle = edgeAngle.getValue();
    Real tAngle = triAngle.getValue();
    Coord dir = direction.getValue();
    Coord norm = normal.getValue();

    if (eAngle>0)
        dir.normalize();

    if (tAngle>0)
        norm.normalize();

    SetIndex& pointIndices = *(f_pointIndices.beginEdit());
    SetEdge& edgeIndices = *(f_edgeIndices.beginEdit());
    SetTriangle& triangleIndices = *(f_triangleIndices.beginEdit());

    triangleIndices.clear();
    pointIndices.clear();
    edgeIndices.clear();

    x0 = &f_X0.getValue();

    const BaseMeshTopology::SeqEdges* edges = &f_edges.getValue();
    const BaseMeshTopology::SeqTriangles* triangles = &f_triangles.getValue();

    if (c.size() == r.size())
    {
        //points
        for(unsigned int i=0; i<pointSize; ++i)
        {
            for (unsigned int j=0; j<c.size(); ++j)
                if (containsPoint(c[j], r[j], (*x0)[i]))
                    pointIndices.push_back(i);
        }
        //edges
        for(unsigned int i=0; i<edges->size(); ++i)
        {
            const BaseMeshTopology::Edge& edge = (*edges)[i];
            bool inside = false;
            for (unsigned int j=0; j<c.size(); ++j)
                if (containsEdge(c[j], r[j], edge)) inside = true;
            if(inside)
            {
                if (eAngle > 0)
                {
                    Coord n = (*x0)[edge[1]]-(*x0)[edge[0]];
                    n.normalize();
                    if (fabs(dot(n,dir)) < fabs(cos(eAngle*M_PI/180.0))) continue;
                }
                edgeIndices.push_back(i);
            }
        }

        //triangles
        for(unsigned int i=0; i<triangles->size(); ++i)
        {
            const BaseMeshTopology::Triangle& triangle = (*triangles)[i];
            bool inside = false;
            for (unsigned int j=0; j<c.size(); ++j)
                if (containsTriangle(c[j], r[j], triangle)) inside = true;
            if (inside)
            {
                if (tAngle > 0)
                {
                    Coord n = cross((*x0)[triangle[2]]-(*x0)[triangle[0]], (*x0)[triangle[1]]-(*x0)[triangle[0]]);
                    n.normalize();
                    if (dot(n,norm) < cos(tAngle*M_PI/180.0)) continue;
                }
                triangleIndices.push_back(i);
            }
        }
    }

    f_pointIndices.endEdit();
    f_edgeIndices.endEdit();
    f_triangleIndices.endEdit();
}

template <class DataTypes>
void PrimitivesInSphereROI<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels() || !isVisible.getValue())
        return;

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        ///draw the boxes
        const helper::vector<Vec3>& c=centers.getValue();
        const helper::vector<Real>& r=radii.getValue();

        for (unsigned int i=0; i<c.size() && i<r.size(); ++i)
        {
            helper::gl::drawWireSphere(c[i], (float)(r[i]/2.0));

            if (edgeAngle.getValue() > 0)
            {
                helper::gl::drawCone(c[i], c[i] + direction.getValue()*(cos(edgeAngle.getValue()*M_PI/180.0)*r[i]), 0, (float)sin(edgeAngle.getValue()*M_PI/180.0)*((float)r[i]));
            }

            if (triAngle.getValue() > 0)
            {
                helper::gl::drawCone(c[i], c[i] + normal.getValue()*(cos(triAngle.getValue()*M_PI/180.0)*r[i]), 0, (float)sin(triAngle.getValue()*M_PI/180.0)*((float)r[i]));
            }
        }
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
