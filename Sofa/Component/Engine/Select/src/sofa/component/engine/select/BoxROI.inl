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
#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <limits>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/helper/accessor.h>

namespace sofa::component::engine::select::boxroi
{

using core::behavior::BaseMechanicalState ;
using core::topology::TopologyContainer ;
using core::topology::BaseMeshTopology ;
using core::objectmodel::ComponentState ;
using core::objectmodel::BaseData ;
using core::objectmodel::Event ;
using core::loader::MeshLoader ;
using core::ExecParams ;
using type::TBoundingBox ;
using type::Vec3 ;
using type::Vec4f ;
using helper::WriteOnlyAccessor ;
using helper::ReadAccessor ;
using type::vector ;

template <class DataTypes>
BoxROI<DataTypes>::BoxROI()
    : d_alignedBoxes( initData(&d_alignedBoxes, "box", "List of boxes defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , d_orientedBoxes( initData(&d_orientedBoxes, "orientedBox", "List of boxes defined by 3 points (p0, p1, p2) and a depth distance \n"
                                "A parallelogram will be defined by (p0, p1, p2, p3 = p0 + (p2-p1)). \n"
                                "The box will finaly correspond to the parallelogram extrusion of depth/2 \n"
                                "along its normal and depth/2 in the opposite direction. ") )
    , d_X0( initData (&d_X0, "position", "Rest position coordinates of the degrees of freedom. \n"
                                         "If empty the positions from a MechanicalObject then a MeshLoader are searched in the current context. \n"
                                         "If none are found the parent's context is searched for MechanicalObject." ) )
    , d_edges(initData (&d_edges, "edges", "Edge Topology") )
    , d_triangles(initData (&d_triangles, "triangles", "Triangle Topology") )
    , d_tetrahedra(initData (&d_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , d_hexahedra(initData (&d_hexahedra, "hexahedra", "Hexahedron Topology") )
    , d_quad(initData (&d_quad, "quad", "Quad Topology") )
    , d_computeEdges( initData(&d_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI. (default = true)") )
    , d_computeTriangles( initData(&d_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI. (default = true)") )
    , d_computeTetrahedra( initData(&d_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI. (default = true)") )
    , d_computeHexahedra( initData(&d_computeHexahedra, true,"computeHexahedra","If true, will compute hexahedra list and index list inside the ROI. (default = true)") )
    , d_computeQuad( initData(&d_computeQuad, true,"computeQuad","If true, will compute quad list and index list inside the ROI. (default = true)") )
    , d_strict( initData(&d_strict, true,"strict","If true, an element is inside the box iif all of its nodes are inside. If False, only the center point of the element is checked. (default = true)") )
    , d_indices( initData(&d_indices,"indices","Indices of the points contained in the ROI") )
    , d_edgeIndices( initData(&d_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , d_triangleIndices( initData(&d_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , d_tetrahedronIndices( initData(&d_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , d_hexahedronIndices( initData(&d_hexahedronIndices,"hexahedronIndices","Indices of the hexahedra contained in the ROI") )
    , d_quadIndices( initData(&d_quadIndices,"quadIndices","Indices of the quad contained in the ROI") )
    , d_pointsInROI( initData(&d_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , d_edgesInROI( initData(&d_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , d_trianglesInROI( initData(&d_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , d_tetrahedraInROI( initData(&d_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , d_hexahedraInROI( initData(&d_hexahedraInROI,"hexahedraInROI","Hexahedra contained in the ROI") )
    , d_quadInROI( initData(&d_quadInROI,"quadInROI","Quad contained in the ROI") )
    , d_nbIndices( initData(&d_nbIndices,"nbIndices", "Number of selected indices") )
    , d_drawBoxes( initData(&d_drawBoxes,false,"drawBoxes","Draw Boxes. (default = false)") )
    , d_drawPoints( initData(&d_drawPoints,false,"drawPoints","Draw Points. (default = false)") )
    , d_drawEdges( initData(&d_drawEdges,false,"drawEdges","Draw Edges. (default = false)") )
    , d_drawTriangles( initData(&d_drawTriangles,false,"drawTriangles","Draw Triangles. (default = false)") )
    , d_drawTetrahedra( initData(&d_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra. (default = false)") )
    , d_drawHexahedra( initData(&d_drawHexahedra,false,"drawHexahedra","Draw Tetrahedra. (default = false)") )
    , d_drawQuads( initData(&d_drawQuads,false,"drawQuads","Draw Quads. (default = false)") )
    , d_drawSize( initData(&d_drawSize,1.0,"drawSize","rendering size for box and topological elements") )
    , d_doUpdate( initData(&d_doUpdate,(bool)true,"doUpdate","If true, updates the selection at the beginning of simulation steps. (default = true)") )

    /// In case you add a new attribute please also add it into to the BoxROI_test.cpp::attributesTests
    /// In case you want to remove or rename an attribute, please keep it as-is but add a warning message
    /// using msg_warning saying to the user of this component that the attribute is deprecated and solutions/replacement
    /// he has to fix his scene.
{
    sofa::helper::getWriteOnlyAccessor(d_indices).push_back(0);

    addAlias(&d_pointsInROI,"pointsInBox");
    addAlias(&d_edgesInROI,"edgesInBox");
    addAlias(&d_trianglesInROI,"f_trianglesInBox");
    addAlias(&d_tetrahedraInROI,"f_tetrahedraInBox");
    addAlias(&d_hexahedraInROI,"f_tetrahedraInBox");
    addAlias(&d_quadInROI,"f_quadInBOX");

    addInput(&d_alignedBoxes);
    addInput(&d_orientedBoxes);

    addInput(&d_X0);
    addInput(&d_edges);
    addInput(&d_triangles);
    addInput(&d_tetrahedra);
    addInput(&d_hexahedra);
    addInput(&d_quad);

    addOutput(&d_indices);
    addOutput(&d_edgeIndices);
    addOutput(&d_triangleIndices);
    addOutput(&d_tetrahedronIndices);
    addOutput(&d_hexahedronIndices);
    addOutput(&d_quadIndices);
    addOutput(&d_pointsInROI);
    addOutput(&d_edgesInROI);
    addOutput(&d_trianglesInROI);
    addOutput(&d_tetrahedraInROI);
    addOutput(&d_hexahedraInROI);
    addOutput(&d_quadInROI);
    addOutput(&d_nbIndices);
}

template <class DataTypes>
void BoxROI<DataTypes>::init()
{
    /// If the position attribute is not set we are trying to
    /// automatically load the positions from the current context MechanicalState if any, then
    /// in a MeshLoad if any and in case of failure it will finally search it in the parent's
    /// context.
    if (!d_X0.isSet())
    {
        msg_info(this) << "No attribute 'position' set.\n"
                          "Searching in the context for a MechanicalObject or MeshLoader.\n" ;

        BaseMechanicalState* mstate = nullptr ;
        this->getContext()->get(mstate, BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                d_X0.setParent(parent);
                d_X0.setReadOnly(true);
            }else{
                msg_warning(this) << "No attribute 'rest_position' in component '" << getName() << "'.\n"
                                  << "The BoxROI component thus have no input and is thus deactivated.\n" ;
                d_componentState.setValue(ComponentState::Invalid) ;
                return ;
            }
        }
        else
        {
            MeshLoader* loader = nullptr ;
            this->getContext()->get(loader,BaseContext::Local);
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    d_X0.setParent(parent);
                    d_X0.setReadOnly(true);
                }else{
                    msg_warning(this) << "No attribute 'position' in component '" << getName() << "'.\n"
                                      << "The BoxROI component thus have no input and is thus deactivated.\n" ;
                    d_componentState.setValue(ComponentState::Invalid) ;
                    return ;
                }
            }
            else   // no local state, no loader => find upward
            {
                this->getContext()->get(mstate, BaseContext::SearchUp);
                if(!mstate){
                    msg_error(this) <<  "Unable to find a MechanicalObject for this component. "
                                        "To remove this error message you can either:\n"
                                        "   - to specifiy the DOF where to apply the BoxROI with the 'position' attribute.\n"
                                        "   - to add MechanicalObject or MeshLoader component before the BoxROI in the scene graph.\n";
                    d_componentState.setValue(ComponentState::Invalid) ;
                    return ;
                }

                BaseData* parent = mstate->findData("rest_position");
                if(!parent){
                    dmsg_error(this) <<  "Unable to find a rest_position attribute in the MechanicalObject '" << mstate->getName() << "'";
                    d_componentState.setValue(ComponentState::Invalid) ;
                    return ;
                }
                d_X0.setParent(parent);
                d_X0.setReadOnly(true);
            }
        }
    }


    if (!d_edges.isSet() || !d_triangles.isSet() || !d_tetrahedra.isSet() || !d_hexahedra.isSet() || !d_quad.isSet() )
    {
        msg_info(this) << "No topology given. Searching for a TopologyContainer and a BaseMeshTopology in the current context.\n";

        TopologyContainer* topologyContainer;
        this->getContext()->get(topologyContainer,BaseContext::Local);

        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local);

        if (topologyContainer || topology)
        {
            if (!d_edges.isSet() && d_computeEdges.getValue())
            {
                BaseData* eparent = topologyContainer?topologyContainer->findData("edges"):topology->findData("edges");
                if (eparent)
                {
                    d_edges.setParent(eparent);
                    d_edges.setReadOnly(true);
                }
            }
            if (!d_triangles.isSet() && d_computeTriangles.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("triangles"):topology->findData("triangles");
                if (tparent)
                {
                    d_triangles.setParent(tparent);
                    d_triangles.setReadOnly(true);
                }
            }
            if (!d_tetrahedra.isSet() && d_computeTetrahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("tetrahedra"):topology->findData("tetrahedra");
                if (tparent)
                {
                    d_tetrahedra.setParent(tparent);
                    d_tetrahedra.setReadOnly(true);
                }
            }
            if (!d_hexahedra.isSet() && d_computeHexahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("hexahedra"):topology->findData("hexahedra");
                if (tparent)
                {
                    d_hexahedra.setParent(tparent);
                    d_hexahedra.setReadOnly(true);
                }
            }
            if (!d_quad.isSet() && d_computeQuad.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("quads"):topology->findData("quads");
                if (tparent)
                {
                    d_quad.setParent(tparent);
                    d_quad.setReadOnly(true);
                }
            }
        }/*else{
            msg_warning(this) << "No primitives provided nor TopologyContainer and a BaseMeshTopology in the current context.\n"
                                 "To remove this message you can either: \n"
                                 "  - set value into one or more of the attributes 'edges', 'triangles', 'tetrahedra', 'hexahedra'. \n"
                                 "  - add a TopologyContainer and a BaseMeshTopology in the context of this object. \n";
            d_componentState.setValue(ComponentState::Invalid) ;
            return ;
        }*/
    }


    d_componentState.setValue(ComponentState::Valid) ;

    /// The following is a trick to force the initial selection of the element by the engine.
    const bool tmp=d_doUpdate.getValue() ;
    d_doUpdate.setValue(true);
    setDirtyValue();
    if(!d_alignedBoxes.isSet() && !d_orientedBoxes.isSet())
    {
        auto alignedBoxes = sofa::helper::getWriteOnlyAccessor(d_alignedBoxes);
        alignedBoxes.push_back(type::Vec6(0,0,0,1,1,1));
    }

    auto alignedBoxes = sofa::helper::getWriteOnlyAccessor(d_alignedBoxes);
    if (!alignedBoxes.empty())
    {
        for (unsigned int bi=0; bi<alignedBoxes.size(); ++bi)
        {
            if (alignedBoxes[bi][0] > alignedBoxes[bi][3]) std::swap(alignedBoxes[bi][0], alignedBoxes[bi][3]);
            if (alignedBoxes[bi][1] > alignedBoxes[bi][4]) std::swap(alignedBoxes[bi][1], alignedBoxes[bi][4]);
            if (alignedBoxes[bi][2] > alignedBoxes[bi][5]) std::swap(alignedBoxes[bi][2], alignedBoxes[bi][5]);
        }
    }

    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        static const std::string message = "\nOriented bounding boxes are not supported in " + std::to_string(DataTypes::spatial_dimensions) + "D";
        d_orientedBoxes.setHelp(d_orientedBoxes.getHelp() + message);
        msg_warning_when(d_orientedBoxes.isSet()) << message;
    }

    computeOrientedBoxes();

    update();
    d_doUpdate.setValue(tmp);
}

template <class DataTypes>
void BoxROI<DataTypes>::reinit()
{
    init();
}


template <class DataTypes>
void BoxROI<DataTypes>::computeOrientedBoxes()
{
    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        return;
    }

    const vector<Vec10>& orientedBoxes = d_orientedBoxes.getValue();

    if(orientedBoxes.empty())
        return;

    m_orientedBoxes.resize(orientedBoxes.size());

    for(unsigned int i=0; i<orientedBoxes.size(); i++)
    {
        const Vec10& box = orientedBoxes[i];

        const type::Vec3 p0 = type::Vec3(box[0], box[1], box[2]);
        const type::Vec3 p1 = type::Vec3(box[3], box[4], box[5]);
        const type::Vec3 p2 = type::Vec3(box[6], box[7], box[8]);
        double depth = box[9];

        type::Vec3 normal = (p1-p0).cross(p2-p0);
        normal.normalize();

        const type::Vec3 p3 = p0 + (p2-p1);
        const type::Vec3 p6 = p2 + normal * depth;

        type::Vec3 plane0 = (p1-p0).cross(normal);
        plane0.normalize();

        type::Vec3 plane1 = (p2-p3).cross(p6-p3);
        plane1.normalize();

        type::Vec3 plane2 = (p3-p0).cross(normal);
        plane2.normalize();

        type::Vec3 plane3 = (p2-p1).cross(p6-p2);
        plane3.normalize();


        m_orientedBoxes[i].p0 = p0;
        m_orientedBoxes[i].p2 = p2;
        m_orientedBoxes[i].normal = normal;
        m_orientedBoxes[i].plane0 = plane0;
        m_orientedBoxes[i].plane1 = plane1;
        m_orientedBoxes[i].plane2 = plane2;
        m_orientedBoxes[i].plane3 = plane3;
        m_orientedBoxes[i].width = fabs(dot((p2-p0),plane0));
        m_orientedBoxes[i].length = fabs(dot((p2-p0),plane2));
        m_orientedBoxes[i].depth = depth;
    }
}


template <class DataTypes>
bool BoxROI<DataTypes>::isPointInOrientedBox(const typename DataTypes::CPos& point, const OrientedBox& box)
{
    if constexpr (DataTypes::spatial_dimensions != 3)
    {
        return false;
    }
    else
    {
        const type::Vec3 pv0 = type::Vec3(point[0]-box.p0[0], point[1]-box.p0[1], point[2]-box.p0[2]);
        const type::Vec3 pv1 = type::Vec3(point[0]-box.p2[0], point[1]-box.p2[1], point[2]-box.p2[2]);

        if( fabs(dot(pv0, box.plane0)) <= box.width && fabs(dot(pv1, box.plane1)) <= box.width )
        {
            if ( fabs(dot(pv0, box.plane2)) <= box.length && fabs(dot(pv1, box.plane3)) <= box.length )
            {
                if ( !(fabs(dot(pv0, box.normal)) <= fabs(box.depth/2)) )
                    return false;
            }
            else
                return false;
        }
        else
            return false;

        return true;
    }
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInAlignedBox(const typename DataTypes::CPos& p, const type::Vec6& box)
{
    static_assert(std::is_same_v<typename DataTypes::CPos::size_type, typename type::Vec6::size_type>);

    for (typename type::Vec6::size_type i = 0; i < DataTypes::spatial_dimensions; ++i)
    {
        if (p[i] < box[i] || p[i] > box[i + 3])
        {
            return false;
        }
    }
    return true;
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInBoxes(const typename DataTypes::CPos& p)
{
    const vector<type::Vec6>& alignedBoxes = d_alignedBoxes.getValue();

    for (unsigned int i=0; i<alignedBoxes.size(); ++i)
        if (isPointInAlignedBox(p, alignedBoxes[i]))
            return true;

    if constexpr (DataTypes::spatial_dimensions == 3)
    {
        for (unsigned int i=0; i<m_orientedBoxes.size(); ++i)
            if (isPointInOrientedBox(p, m_orientedBoxes[i]))
                return true;
    }

    return false;
}

template <class DataTypes>
bool BoxROI<DataTypes>::isPointInBoxes(const PointID& pid)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p =  DataTypes::getCPos(x0[pid]);
    return ( isPointInBoxes(p) );
}

template <class DataTypes>
bool BoxROI<DataTypes>::isEdgeInBoxes(const Edge& e)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[e[0]]);
    CPos p1 =  DataTypes::getCPos(x0[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInBoxes(c);
}

template <class DataTypes>
bool BoxROI<DataTypes>::isEdgeInBoxesStrict(const Edge& e)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[e[0]]);
    CPos p1 =  DataTypes::getCPos(x0[e[1]]);

    return isPointInBoxes(p0) && isPointInBoxes(p1);
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTriangleInBoxes(const Triangle& t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInBoxes(c));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTriangleInBoxesStrict(const Triangle& t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);

    return (isPointInBoxes(p0) && isPointInBoxes(p1) && isPointInBoxes(p2));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTetrahedronInBoxes(const Tetra &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBoxes(c));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTetrahedronInBoxesStrict(const Tetra &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);

    return (isPointInBoxes(p0) && isPointInBoxes(p1) && isPointInBoxes(p2) && isPointInBoxes(p3));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isHexahedronInBoxes(const Hexa &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos p4 =  DataTypes::getCPos(x0[t[4]]);
    CPos p5 =  DataTypes::getCPos(x0[t[5]]);
    CPos p6 =  DataTypes::getCPos(x0[t[6]]);
    CPos p7 =  DataTypes::getCPos(x0[t[7]]);
    CPos c = (p7+p6+p5+p4+p3+p2+p1+p0)/8.0;

    return (isPointInBoxes(c));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isHexahedronInBoxesStrict(const Hexa &t)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos p4 =  DataTypes::getCPos(x0[t[4]]);
    CPos p5 =  DataTypes::getCPos(x0[t[5]]);
    CPos p6 =  DataTypes::getCPos(x0[t[6]]);
    CPos p7 =  DataTypes::getCPos(x0[t[7]]);

    return (isPointInBoxes(p0) && isPointInBoxes(p1) && isPointInBoxes(p2) && isPointInBoxes(p3) &&
            isPointInBoxes(p4) && isPointInBoxes(p5) && isPointInBoxes(p6) && isPointInBoxes(p7));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isQuadInBoxes(const Quad& q)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[q[0]]);
    CPos p1 =  DataTypes::getCPos(x0[q[1]]);
    CPos p2 =  DataTypes::getCPos(x0[q[2]]);
    CPos p3 =  DataTypes::getCPos(x0[q[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBoxes(c));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isQuadInBoxesStrict(const Quad& q)
{
    const VecCoord& x0 = d_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[q[0]]);
    CPos p1 =  DataTypes::getCPos(x0[q[1]]);
    CPos p2 =  DataTypes::getCPos(x0[q[2]]);
    CPos p3 =  DataTypes::getCPos(x0[q[3]]);

    return (isPointInBoxes(p0) && isPointInBoxes(p1) && isPointInBoxes(p2) && isPointInBoxes(p3));
}

// The update method is called when the engine is marked as dirty.
template <class DataTypes>
void BoxROI<DataTypes>::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid){
        return ;
    }


    if(d_doUpdate.getValue()){

        // Check whether an element can partially be inside the box or if all of its nodes must be inside
        const bool strict = d_strict.getValue();

        // Write accessor for topological element indices in BOX
        SetIndex& indices = *d_indices.beginWriteOnly();
        SetIndex& edgeIndices = *d_edgeIndices.beginWriteOnly();
        SetIndex& triangleIndices = *d_triangleIndices.beginWriteOnly();
        SetIndex& tetrahedronIndices = *d_tetrahedronIndices.beginWriteOnly();
        SetIndex& hexahedronIndices = *d_hexahedronIndices.beginWriteOnly();
        SetIndex& quadIndices = *d_quadIndices.beginWriteOnly();

        // Write accessor for toplogical element in BOX
        WriteOnlyAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
        WriteOnlyAccessor< Data<vector<Edge> > > edgesInROI = d_edgesInROI;
        WriteOnlyAccessor< Data<vector<Triangle> > > trianglesInROI = d_trianglesInROI;
        WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
        WriteOnlyAccessor< Data<vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
        WriteOnlyAccessor< Data<vector<Quad> > > quadInROI = d_quadInROI;

        // Clear lists
        indices.clear();
        edgeIndices.clear();
        triangleIndices.clear();
        tetrahedronIndices.clear();
        hexahedronIndices.clear();
        quadIndices.clear();


        pointsInROI.clear();
        edgesInROI.clear();
        trianglesInROI.clear();
        tetrahedraInROI.clear();
        hexahedraInROI.clear();
        quadInROI.clear();


        if (d_X0.getValue().size() == 0)
        {
            msg_warning() << "No rest position yet defined. Box might not work properly. \n"
                            "This may be caused by an early call of init() on the box before  \n"
                            "the mesh or the MechanicalObject of the node was initialized too";
            return;
        }

        const vector<type::Vec6>&  alignedBoxes  = d_alignedBoxes.getValue();
        const vector<Vec10>& orientedBoxes = d_orientedBoxes.getValue();

        if (alignedBoxes.empty() && orientedBoxes.empty()) { return; }


        // Read accessor for input topology
        const ReadAccessor< Data<vector<Edge> > > edges = d_edges;
        const ReadAccessor< Data<vector<Triangle> > > triangles = d_triangles;
        const ReadAccessor< Data<vector<Tetra> > > tetrahedra = d_tetrahedra;
        const ReadAccessor< Data<vector<Hexa> > > hexahedra = d_hexahedra;
        const ReadAccessor< Data<vector<Quad> > > quad = d_quad;

        const VecCoord& x0 = d_X0.getValue();


        //Points
        for( unsigned i=0; i<x0.size(); ++i )
        {
            if (isPointInBoxes(i))
            {
                indices.push_back(i);
                pointsInROI.push_back(x0[i]);
            }
        }

        //Edges
        if (d_computeEdges.getValue())
        {
            for(unsigned int i=0 ; i<edges.size() ; i++)
            {
                Edge e = edges[i];
                const bool is_in_box = (strict) ? isEdgeInBoxesStrict(e) : isEdgeInBoxes(e);
                if (is_in_box)
                {
                    edgeIndices.push_back(i);
                    edgesInROI.push_back(e);
                }
            }
        }

        //Triangles
        if (d_computeTriangles.getValue())
        {
            for(unsigned int i=0 ; i<triangles.size() ; i++)
            {
                Triangle t = triangles[i];
                const bool is_in_box = (strict) ? isTriangleInBoxesStrict(t) : isTriangleInBoxes(t);
                if (is_in_box)
                {
                    triangleIndices.push_back(i);
                    trianglesInROI.push_back(t);
                }
            }
        }

        //Tetrahedra
        if (d_computeTetrahedra.getValue())
        {
            for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
            {
                Tetra t = tetrahedra[i];
                const bool is_in_box = (strict) ? isTetrahedronInBoxesStrict(t) : isTetrahedronInBoxes(t);
                if (is_in_box)
                {
                    tetrahedronIndices.push_back(i);
                    tetrahedraInROI.push_back(t);
                }
            }
        }

        //Hexahedra
        if (d_computeHexahedra.getValue())
        {
            for(unsigned int i=0 ; i<hexahedra.size() ; i++)
            {
                Hexa t = hexahedra[i];
                const bool is_in_box = (strict) ? isHexahedronInBoxesStrict(t) : isHexahedronInBoxes(t);
                if (is_in_box)
                {
                    hexahedronIndices.push_back(i);
                    hexahedraInROI.push_back(t);
                }
            }
        }

        //Quads
        if (d_computeQuad.getValue())
        {
            for(unsigned int i=0 ; i<quad.size() ; i++)
            {
                Quad q = quad[i];
                const bool is_in_box = (strict) ? isQuadInBoxesStrict(q) : isQuadInBoxes(q);
                if (is_in_box)
                {
                    quadIndices.push_back(i);
                    quadInROI.push_back(q);
                }
            }
        }


        d_nbIndices.setValue(sofa::Size(indices.size()));
    }
}


template <class DataTypes>
void BoxROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return ;

    if (!vparams->displayFlags().getShowBehaviorModels() && !this->d_drawSize.getValue())
        return;

    const VecCoord& x0 = d_X0.getValue();
    constexpr auto color = sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f);


    ///draw the boxes
    if( d_drawBoxes.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<type::Vec3> vertices;

        const vector<type::Vec6>&  alignedBoxes =d_alignedBoxes.getValue();
        const vector<Vec10>& orientedBoxes=d_orientedBoxes.getValue();

        for (unsigned int bi=0; bi<alignedBoxes.size(); ++bi)
        {
            const type::Vec6& b=alignedBoxes[bi];
            const Real& Xmin=b[0];
            const Real& Xmax=b[3];
            const Real& Ymin=b[1];
            const Real& Ymax=b[4];
            const Real& Zmin=b[2];
            const Real& Zmax=b[5];
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmax) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmax) );
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmin,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmin,Ymax,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmax) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmax,Ymin,Zmin) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmin) );
            vertices.push_back( type::Vec3(Xmax,Ymax,Zmax) );
            vparams->drawTool()->drawLines(vertices, linesWidth , color );
        }

        for (unsigned int bi=0; bi<orientedBoxes.size(); ++bi)
        {
            const Vec10& box=orientedBoxes[bi];

            vector<type::Vec3> points;
            points.resize(8);
            getPointsFromOrientedBox(box, points);

            vertices.push_back( points[0] );
            vertices.push_back( points[1] );
            vertices.push_back( points[0] );
            vertices.push_back( points[4] );
            vertices.push_back( points[0] );
            vertices.push_back( points[3] );

            vertices.push_back( points[2] );
            vertices.push_back( points[1] );
            vertices.push_back( points[2] );
            vertices.push_back( points[6] );
            vertices.push_back( points[2] );
            vertices.push_back( points[3] );

            vertices.push_back( points[6] );
            vertices.push_back( points[7] );
            vertices.push_back( points[6] );
            vertices.push_back( points[5] );

            vertices.push_back( points[4] );
            vertices.push_back( points[5] );
            vertices.push_back( points[4] );
            vertices.push_back( points[7] );

            vertices.push_back( points[1] );
            vertices.push_back( points[5] );
            vertices.push_back( points[3] );
            vertices.push_back( points[7] );
            vparams->drawTool()->drawLines(vertices, linesWidth , color );
        }
    }

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    ///draw points in ROI
    if( d_drawPoints.getValue())
    {
        float pointsWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            type::Vec3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawPoints(vertices, pointsWidth, color);
    }

    ///draw edges in ROI
    if( d_drawEdges.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Edge> > > edgesInROI = d_edgesInROI;
        for (unsigned int i=0; i<edgesInROI.size() ; ++i)
        {
            Edge e = edgesInROI[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[e[j]]);
                type::Vec3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw triangles in ROI
    if( d_drawTriangles.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Triangle> > > trianglesInROI = d_trianglesInROI;
        for (unsigned int i=0; i<trianglesInROI.size() ; ++i)
        {
            Triangle t = trianglesInROI[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                type::Vec3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawTriangles(vertices, color);
    }

    ///draw tetrahedra in ROI
    if( d_drawTetrahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
        for (unsigned int i=0; i<tetrahedraInROI.size() ; ++i)
        {
            Tetra t = tetrahedraInROI[i];
            for (unsigned int j=0 ; j<4 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                type::Vec3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            type::Vec3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[2]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[1]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[3]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw hexahedra in ROI
    if( d_drawHexahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
        for (unsigned int i=0; i<hexahedraInROI.size() ; ++i)
        {
            Hexa t = hexahedraInROI[i];
            for (unsigned int j=0 ; j<8 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                type::Vec3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            type::Vec3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[2]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[1]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[3]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[4]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[5]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[6]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
            p = DataTypes::getCPos(x0[t[7]]);
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw quads in ROI
    if( d_drawQuads.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = d_drawSize.getValue() ? (float)d_drawSize.getValue() : 1;
        std::vector<type::Vec3> vertices;
        ReadAccessor<Data<vector<Quad> > > quadsInROI = d_quadInROI;
        for (unsigned i=0; i<quadsInROI.size(); ++i)
        {
            Quad q = quadsInROI[i];
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[j]]);
                type::Vec3 pv;
                for (unsigned k=0; k<max_spatial_dimensions; k++)
                    pv[k] = p[k];
                vertices.push_back(pv);
            }
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[(j+1)%4]]);
                type::Vec3 pv;
                for (unsigned k=0; k<max_spatial_dimensions; k++)
                    pv[k] = p[k];
                vertices.push_back(pv);
            }

        }
        vparams->drawTool()->drawLines(vertices,linesWidth,color);
    }

}


template <class DataTypes>
void BoxROI<DataTypes>::computeBBox(const ExecParams*  params , bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( onlyVisible && !d_drawBoxes.getValue() )
        return;

    if(d_componentState.getValue() == ComponentState::Invalid)
        return ;

    const vector<type::Vec6>&  alignedBoxes =d_alignedBoxes.getValue();
    const vector<Vec10>& orientedBoxes=d_orientedBoxes.getValue();

    const Real max_real = std::numeric_limits<Real>::max();
    const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (unsigned int bi=0; bi<alignedBoxes.size(); ++bi)
    {
        const type::Vec6& box=alignedBoxes[bi];
        if (box[0] < minBBox[0]) minBBox[0] = box[0];
        if (box[1] < minBBox[1]) minBBox[1] = box[1];
        if (box[2] < minBBox[2]) minBBox[2] = box[2];
        if (box[3] > maxBBox[0]) maxBBox[0] = box[3];
        if (box[4] > maxBBox[1]) maxBBox[1] = box[4];
        if (box[5] > maxBBox[2]) maxBBox[2] = box[5];
    }

    for (unsigned int bi=0; bi<orientedBoxes.size(); ++bi)
    {
        const Vec10& box=orientedBoxes[bi];

        vector<type::Vec3> points;
        points.resize(8);
        getPointsFromOrientedBox(box, points);

        for(int i=0; i<8; i++)
        {
            if (points[i][0] < minBBox[0]) minBBox[0] = points[i][0];
            if (points[i][1] < minBBox[1]) minBBox[1] = points[i][1];
            if (points[i][2] < minBBox[2]) minBBox[2] = points[i][2];
            if (points[i][0] > maxBBox[0]) maxBBox[0] = points[i][0];
            if (points[i][1] > maxBBox[1]) maxBBox[1] = points[i][1];
            if (points[i][2] > maxBBox[2]) maxBBox[2] = points[i][2];
        }
    }

    this->f_bbox.setValue(TBoundingBox<Real>(minBBox,maxBBox));
}


template <class DataTypes>
void BoxROI<DataTypes>::getPointsFromOrientedBox(const Vec10& box, vector<type::Vec3>& points)
{
    points.resize(8);
    points[0] = type::Vec3(box[0], box[1], box[2]);
    points[1] = type::Vec3(box[3], box[4], box[5]);
    points[2] = type::Vec3(box[6], box[7], box[8]);
    const double depth = box[9];

    type::Vec3 normal = (points[1]-points[0]).cross(points[2]-points[0]);
    normal.normalize();

    points[0] += normal * depth/2;
    points[1] += normal * depth/2;
    points[2] += normal * depth/2;

    points[3] = points[0] + (points[2]-points[1]);
    points[4] = points[0] - normal * depth;
    points[6] = points[2] - normal * depth;
    points[5] = points[1] - normal * depth;
    points[7] = points[3] - normal * depth;
}


template<class DataTypes>
void BoxROI<DataTypes>::handleEvent(Event *event)
{
    SOFA_UNUSED(event);
}

} // namespace sofa::component::engine::select::boxroi
