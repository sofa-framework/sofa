/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/engine/select/MeshROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>

#include <sofa/helper/logging/Messaging.h>

#include <sofa/component/engine/select/BaseROI.inl>

namespace sofa::component::engine::select
{

using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::BaseContext;
using sofa::core::topology::BaseMeshTopology;
using sofa::core::behavior::BaseMechanicalState;
using core::loader::MeshLoader;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteOnlyAccessor;
using sofa::type::vector;
using core::visual::VisualParams;

template <class DataTypes>
MeshROI<DataTypes>::MeshROI()
    : Inherit()
    , d_roiPositions(initData(&d_roiPositions, "ROIposition", "ROI position coordinates of the degrees of freedom"))
    , d_roiEdges(initData (&d_roiEdges, "ROIedges", "ROI Edge Topology") )
    , d_roiTriangles(initData (&d_roiTriangles, "ROItriangles", "ROI Triangle Topology") )
    , d_computeTemplateTriangles(initData(&d_computeTemplateTriangles, true, "computeMeshROI", "Compute with the mesh (not only bounding box)"))
    , d_box( initData(&d_box, "box", "Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , d_drawOut( initData(&d_drawOut,false,"drawOut","Draw the data not contained in the ROI") )
    , d_drawBox( initData(&d_drawBox,false,"drawBox","Draw the Bounding box around the mesh used for the ROI") )
{
    this->addInput(&d_roiPositions);
    this->addInput(&d_roiEdges);
    this->addInput(&d_roiTriangles);

    this->addOutput(&this->d_box);

    // SOFA_ATTRIBUTE_DEPRECATED("24.12", "25.06", "default behavior is now strict.")
    if (!this->d_strict.isSet())
    {
        // Strict mode was not handled before v24.12, and default behavior was non-strict by itself.
        // so this emulates a default value to false.
        this->d_strict.setValue(false);
    }

    d_X0_i.setOriginalData(&d_roiPositions);
    d_edges_i.setOriginalData(&d_roiEdges);
    d_triangles_i.setOriginalData(&d_roiTriangles);

}

template <class DataTypes>
void MeshROI<DataTypes>::roiInit()
{
    checkInputData();
    computeBoundingBox();
}


template <class DataTypes>
void MeshROI<DataTypes>::checkInputData()
{
    // ROI Mesh init
    if (!d_roiPositions.isSet())
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
                d_roiPositions.setParent(parent);
                d_roiPositions.setReadOnly(true);
            }
        }
        else
        {
            MeshLoader* loader = nullptr;
            this->getContext()->get(loader,BaseContext::Local); // perso
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    d_roiPositions.setParent(parent);
                    d_roiPositions.setReadOnly(true);
                }
            }
        }
    }

    if (!d_roiEdges.isSet() || !d_roiTriangles.isSet() )
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local); // perso
        if (topology)
        {
            if (!d_roiEdges.isSet() && this->d_computeEdges.getValue())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    d_roiEdges.setParent(eparent);
                    d_roiEdges.setReadOnly(true);
                }
            }
            if (!d_roiTriangles.isSet() && this->d_computeTriangles.getValue())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    d_roiTriangles.setParent(tparent);
                    d_roiTriangles.setReadOnly(true);
                }
            }
        }
    }
}


template <class DataTypes>
void MeshROI<DataTypes>::computeBoundingBox()
{
    // Bounding Box computation
    type::Vec6 b = d_box.getValue();
    ReadAccessor<Data<VecCoord>> points_i = d_roiPositions;
    if(points_i.size()>0)
    {
        const CPos& p = DataTypes::getCPos(points_i[0]);
        b[0] = p[0]; b[1] = p[1]; b[2] = p[2];
        b[3] = p[0]; b[4] = p[1]; b[5] = p[2];
        for (unsigned int i=1; i<points_i.size() ; ++i)
        {
            const CPos& q = DataTypes::getCPos(points_i[i]);
            if (b[0] < q[0]) b[0] = q[0];
            if (b[1] < q[1]) b[1] = q[1];
            if (b[2] < q[2]) b[2] = q[2];
            if (b[3] > q[0]) b[3] = q[0];
            if (b[4] > q[1]) b[4] = q[1];
            if (b[5] > q[2]) b[5] = q[2];
        }
    }
    if (b[0] > b[3]) std::swap(b[0],b[3]);
    if (b[1] > b[4]) std::swap(b[1],b[4]);
    if (b[2] > b[5]) std::swap(b[2],b[5]);
    d_box.setValue(b);

    msg_info() << "Bounding Box " << b;
}

template <class DataTypes>
bool MeshROI<DataTypes>::checkSameOrder(const CPos& A, const CPos& B, const CPos& pt, const CPos& N) const
{
    CPos vectorial;
    vectorial[0] = (((B[1] - A[1])*(pt[2] - A[2])) - ((pt[1] - A[1])*(B[2] - A[2])));
    vectorial[1] = (((B[2] - A[2])*(pt[0] - A[0])) - ((pt[2] - A[2])*(B[0] - A[0])));
    vectorial[2] = (((B[0] - A[0])*(pt[1] - A[1])) - ((pt[0] - A[0])*(B[1] - A[1])));
    if( (vectorial[0]*N[0] + vectorial[1]*N[1] + vectorial[2]*N[2]) < 0) return false;
    else return true;
}


template <class DataTypes>
bool MeshROI<DataTypes>::isPointInROI(const CPos& p) const
{
    if(!d_computeTemplateTriangles.getValue()) return true;

    if(isPointInBoundingBox(p))
    {
        // Compute the reference point outside the bounding box
        const auto& b = d_box.getValue();
        CPos Vec;
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

        const ReadAccessor< Data<vector<Triangle> > > triangles_i = d_roiTriangles;
        const VecCoord& x0 = d_roiPositions.getValue();
        int Through=0;
        double d=0.0;
        for (unsigned int i=0; i<triangles_i.size() ; ++i)
        {
            Triangle t = triangles_i[i];
            const CPos& p0 =  DataTypes::getCPos(x0[t[0]]);
            const CPos& p1 =  DataTypes::getCPos(x0[t[1]]);
            const CPos& p2 =  DataTypes::getCPos(x0[t[2]]);
            // Normal N compuation of the ROI mesh triangle
            CPos N;
            N[0] = (p1[1]-p0[1])*(p2[2]-p1[2]) - (p1[2]-p0[2])*(p2[1]-p1[1]);
            N[1] = (p1[2]-p0[2])*(p2[0]-p1[0]) - (p1[0]-p0[0])*(p2[2]-p1[2]);
            N[2] = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0]);
            // DotProd computation
            const double DotProd = double (N[0]*Vec[0] + N[1]*Vec[1] + N[2]*Vec[2]);
            if(DotProd !=0)
            {
                // Intersect point with triangle and distance
                d = (N[0]*(p0[0]-p[0])+N[1]*(p0[1]-p[1])+N[2]*(p0[2]-p[2])) / (N[0]*Vec[0]+N[1]*Vec[1]+N[2]*Vec[2]);
                // d negative means that line comes beind the triangle ...
                if(d>=0)
                {
                    CPos ptIN{};
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
bool MeshROI<DataTypes>::isPointInIndices(const unsigned int pointId, const SetIndex& indices)
{
    for (unsigned int i = 0; i < indices.size(); i++)
        if (indices[i] == pointId)
            return true;

    return false;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInIndices(const unsigned int pointId) const
{
    auto indices = sofa::helper::getReadAccessor(this->d_indices);
    return isPointInIndices(pointId, indices.ref());
}

template <class DataTypes>
bool MeshROI<DataTypes>::isPointInBoundingBox(const CPos& p) const
{
    const auto& b = d_box.getValue();
    if( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] )
        return true;
    return false;
}

template <class DataTypes, class Element>
bool applyROIFunc(const Element& e, const core::topology::BaseMeshTopology::SetIndex& indices, const std::function<bool(const Element&)>& roiFunc)
{
    for(const auto eid : e)
    {
        if (!MeshROI<DataTypes>::isPointInIndices(eid, indices))
        {
            return roiFunc(e);
        }
    }
    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isEdgeInROI(const Edge& e) const
{
    return applyROIFunc<DataTypes, Edge>(e, this->d_indices.getValue(), [this](auto&& x)
        { return Inherit::isEdgeInROI(std::forward<decltype(x)>(x)); }
    );
}

template <class DataTypes>
bool MeshROI<DataTypes>::isEdgeInStrictROI(const Edge& e) const
{
    for (int i = 0; i < 2; i++)
    {
        if (!isPointInIndices(e[i]))
        {
            return Inherit::isEdgeInStrictROI(e);
        }
    }
    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTriangleInROI(const Triangle& t) const
{
    for (int i = 0; i < 3; i++)
    {
        if (!isPointInIndices(t[i]))
        {
            return Inherit::isTriangleInROI(t);
        }
    }
    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTriangleInStrictROI(const Triangle& t) const
{
    for (int i = 0; i < 3; i++)
    {
        if (!isPointInIndices(t[i]))
        {
            return Inherit::isTriangleInStrictROI(t);
        }
    }
    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isQuadInROI(const Quad& q) const
{
    for (int i = 0; i < 4; i++)
    {
        if (!isPointInIndices(q[i]))
        {
            return Inherit::isQuadInROI(q);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isQuadInStrictROI(const Quad& q) const
{
    for (int i = 0; i < 4; i++)
    {
        if (!isPointInIndices(q[i]))
        {
            return Inherit::isQuadInStrictROI(q);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTetrahedronInROI(const Tetra& t) const
{
    for (int i = 0; i < 4; i++)
    {
        if (!isPointInIndices(t[i]))
        {
            return Inherit::isTetrahedronInROI(t);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isTetrahedronInStrictROI(const Tetra& t) const
{
    for (int i = 0; i < 4; i++)
    {
        if (!isPointInIndices(t[i]))
        {
            return Inherit::isTetrahedronInStrictROI(t);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isHexahedronInROI(const Hexa& h) const
{
    for (int i = 0; i < 8; i++)
    {
        if (!isPointInIndices(h[i]))
        {
            return Inherit::isHexahedronInROI(h);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::isHexahedronInStrictROI(const Hexa& h) const
{
    for (int i = 0; i < 8; i++)
    {
        if (!isPointInIndices(h[i]))
        {
            return Inherit::isHexahedronInStrictROI(h);
        }
    }

    return true;
}

template <class DataTypes>
bool MeshROI<DataTypes>::roiDoUpdate()
{
    return true;
}

template <class DataTypes>
void MeshROI<DataTypes>::roiDraw(const VisualParams* vparams)
{
    std::vector<sofa::type::Vec3> vertices;

    const float drawSize = (this->d_drawSize.getValue() > 1.0f) ? this->d_drawSize.getValue() : 1.0f;

    const VecCoord& x0_i = d_roiPositions.getValue();
    ///draw ROI points
    if(this->d_drawPoints.getValue())
    {
        helper::ReadAccessor< Data<VecCoord > > points_i = d_roiPositions;
        for (unsigned int i=0; i<points_i.size() ; ++i)
        {
            const CPos& p= DataTypes::getCPos(points_i[i]);
            vertices.emplace_back(p[0], p[1], p[2]);
        }

        vparams->drawTool()->drawPoints(vertices, drawSize, sofa::type::RGBAColor(0.4f, 0.4f, 1.0f, 1.0f));
    }
    // draw ROI edges
    if(this->d_drawEdges.getValue())
    {
        vertices.clear();
        const helper::ReadAccessor< Data<type::vector<Edge> > > edges_i = d_roiEdges;
        for (unsigned int i=0; i<edges_i.size() ; ++i)
        {
            Edge e = edges_i[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                const CPos& p = DataTypes::getCPos(x0_i[e[j]]);
                vertices.emplace_back(p[0], p[1], p[2]);
            }
        }
        vparams->drawTool()->drawLines(vertices, drawSize, sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f));
    }
    // draw ROI triangles
    if(this->d_drawTriangles.getValue())
    {
        vertices.clear();
        const helper::ReadAccessor< Data<type::vector<Triangle> > > triangles_i = d_roiTriangles;
        for (unsigned int i=0; i<triangles_i.size() ; ++i)
        {
            Triangle t = triangles_i[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                const CPos& p= (DataTypes::getCPos(x0_i[t[j]]));
                vertices.emplace_back(p[0], p[1], p[2]);
            }
        }
        vparams->drawTool()->drawTriangles(vertices, sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f));
    }
    
    // draw the bounding box
    if( d_drawBox.getValue())
    {
        vertices.clear();
        const auto& b = d_box.getValue();
        const sofa::type::Vec3 minBBox(b[0], b[1], b[2]);
        const sofa::type::Vec3 maxBBox(b[3], b[4], b[5]);

        vparams->drawTool()->setMaterial(sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f));
        vparams->drawTool()->drawBoundingBox(minBBox, maxBBox, drawSize);

    }
}

template <class DataTypes>
void MeshROI<DataTypes>::roiComputeBBox(const core::ExecParams* params, type::BoundingBox& bbox)
{
    SOFA_UNUSED(params);

    const VecCoord& roiPositions = d_roiPositions.getValue();

    for (const auto& p : roiPositions)
    {
        bbox.include(DataTypes::getCPos(p));
    }
}

} //namespace sofa::component::engine::select
