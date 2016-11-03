/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaEngine/BoxROI.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace engine
{

namespace boxroi
{
using simulation::AnimateBeginEvent ;
using core::behavior::BaseMechanicalState ;
using core::topology::TopologyContainer ;
using core::topology::BaseMeshTopology ;
using core::objectmodel::ComponentState ;
using core::objectmodel::BaseData ;
using core::objectmodel::Event ;
using core::loader::MeshLoader ;
using core::ExecParams ;
using defaulttype::TBoundingBox ;
using defaulttype::Vector3 ;
using defaulttype::Vec4f ;
using helper::WriteOnlyAccessor ;
using helper::ReadAccessor ;
using helper::vector ;

template <class DataTypes>
BoxROI<DataTypes>::BoxROI()
    : boxes( initData(&boxes, "box", "Box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom. "
                                         "If empty the positions from a MechanicalObject then a MeshLoader are searched in the current context."
                                         "If none are found the parent's context is searched for MechanicalObject." ) )
    , f_edges(initData (&f_edges, "edges", "Edge Topology") )
    , f_triangles(initData (&f_triangles, "triangles", "Triangle Topology") )
    , f_tetrahedra(initData (&f_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , f_hexahedra(initData (&f_hexahedra, "hexahedra", "Hexahedron Topology") )
    , f_quad(initData (&f_quad, "quad", "Quad Topology") )
    , f_computeEdges( initData(&f_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI. (default = true)") )
    , f_computeTriangles( initData(&f_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI. (default = true)") )
    , f_computeTetrahedra( initData(&f_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI. (default = true)") )
    , f_computeHexahedra( initData(&f_computeHexahedra, true,"computeHexahedra","If true, will compute hexahedra list and index list inside the ROI. (default = true)") )
    , f_computeQuad( initData(&f_computeQuad, true,"computeQuad","If true, will compute quad list and index list inside the ROI. (default = true)") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_edgeIndices( initData(&f_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , f_triangleIndices( initData(&f_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , f_tetrahedronIndices( initData(&f_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , f_hexahedronIndices( initData(&f_hexahedronIndices,"hexahedronIndices","Indices of the hexahedra contained in the ROI") )
    , f_quadIndices( initData(&f_quadIndices,"quadIndices","Indices of the quad contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , f_edgesInROI( initData(&f_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , f_trianglesInROI( initData(&f_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , f_tetrahedraInROI( initData(&f_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , f_hexahedraInROI( initData(&f_hexahedraInROI,"hexahedraInROI","Hexahedra contained in the ROI") )
    , f_quadInROI( initData(&f_quadInROI,"quadInROI","Quad contained in the ROI") )
    , f_nbIndices( initData(&f_nbIndices,"nbIndices", "Number of selected indices") )
    , p_drawBoxes( initData(&p_drawBoxes,false,"drawBoxes","Draw Boxes. (default = false)") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points. (default = false)") )
    , p_drawEdges( initData(&p_drawEdges,false,"drawEdges","Draw Edges. (default = false)") )
    , p_drawTriangles( initData(&p_drawTriangles,false,"drawTriangles","Draw Triangles. (default = false)") )
    , p_drawTetrahedra( initData(&p_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra. (default = false)") )
    , p_drawHexahedra( initData(&p_drawHexahedra,false,"drawHexahedra","Draw Tetrahedra. (default = false)") )
    , p_drawQuads( initData(&p_drawQuads,false,"drawQuads","Draw Quads. (default = false)") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","rendering size for box and topological elements") )
    , p_doUpdate( initData(&p_doUpdate,(bool)true,"doUpdate","If true, updates the selection at the beginning of simulation steps. (default = true)") )

    /// In case you add a new attribute please also add it into to the BoxROI_test.cpp::attributesTests
    /// In case you want to remove or rename an attribute, please keep it as-is but add a warning message
    /// using msg_warning saying to the user of this component that the attribute is deprecated and solutions/replacement
    /// he has to fix his scene.

    /// Deprecated input attributes
    , d_deprecatedX0( initData (&d_deprecatedX0, "rest_position", "(deprecated) Replaced with the attribute 'position'") )
    , d_deprecatedIsVisible( initData(&d_deprecatedIsVisible, false, "isVisible","(deprecated)Replaced with the attribute 'drawBoxes'") )
{
    //Adding alias to handle old BoxROI outputs
    addAlias(&f_pointsInROI,"pointsInBox");
    addAlias(&f_edgesInROI,"edgesInBox");
    addAlias(&f_trianglesInROI,"f_trianglesInBox");
    addAlias(&f_tetrahedraInROI,"f_tetrahedraInBox");
    addAlias(&f_hexahedraInROI,"f_tetrahedraInBox");
    addAlias(&f_quadInROI,"f_quadInBOX");

    /// Display as few as possible the deprecated data.
    d_deprecatedX0.setDisplayed(false);
    d_deprecatedIsVisible.setDisplayed(false);

    boxes.beginEdit()->push_back(Vec6(0,0,0,1,1,1));
    boxes.endEdit();

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void BoxROI<DataTypes>::init()
{
    if(d_deprecatedX0.isSet()){
        msg_error(this) <<  "The field named 'rest_position' is now deprecated.\n"
                              "Using deprecated attributes may results in invalid simulation as well as slow performances. \n"
                              "To remove this error message you need to fix your sofa scene by changing the field's name from 'rest_position' to 'position'.\n" ;
                               f_X0.copyValue(&d_deprecatedX0);
    }

    if(d_deprecatedIsVisible.isSet()){
        msg_error(this) <<  "The field named 'isVisible' is now deprecated.\n"
                              "Using deprecated attributes may results in invalid simulation as well as slow performances. \n"
                              "To remove this error message you need to fix your sofa scene by changing the field's name from 'isVisible' to 'drawBoxes'.\n" ;
                              p_drawBoxes.copyValue(&d_deprecatedIsVisible);
    }

    /// If the position attribute is not set we are trying to
    /// automatically load the positions from the current context MechanicalState if any, then
    /// in a MeshLoad if any and in case of failure it will finally search it in the parent's
    /// context.
    if (!f_X0.isSet())
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
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }else{
                msg_warning(this) << "No attribute 'rest_position' in component '" << getName() << "'.\n"
                                  << "The BoxROI component thus have no input and is thus deactivated.\n" ;
                m_componentstate = ComponentState::Invalid ;
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
                    f_X0.setParent(parent);
                    f_X0.setReadOnly(true);
                }else{
                    msg_warning(this) << "No attribute 'position' in component '" << getName() << "'.\n"
                                      << "The BoxROI component thus have no input and is thus deactivated.\n" ;
                    m_componentstate = ComponentState::Invalid ;
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
                    m_componentstate = ComponentState::Invalid ;
                    return ;
                }

                BaseData* parent = mstate->findData("rest_position");
                if(!parent){
                    dmsg_error(this) <<  "Unable to find a rest_position attribute in the MechanicalObject '" << mstate->getName() << "'";
                    m_componentstate = ComponentState::Invalid ;
                    return ;
                }
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
    }


    if (!f_edges.isSet() || !f_triangles.isSet() || !f_tetrahedra.isSet() || !f_hexahedra.isSet() || !f_quad.isSet() )
    {
        msg_info(this) << "No topology given. Searching for a TopologyContainer and a BaseMeshTopology in the current context.\n";

        TopologyContainer* topologyContainer;
        this->getContext()->get(topologyContainer,BaseContext::Local);

        BaseMeshTopology* topology;
        this->getContext()->get(topology,BaseContext::Local);

        if (topologyContainer || topology)
        {
            if (!f_edges.isSet() && f_computeEdges.getValue())
            {
                BaseData* eparent = topologyContainer?topologyContainer->findData("edges"):topology->findData("edges");
                if (eparent)
                {
                    f_edges.setParent(eparent);
                    f_edges.setReadOnly(true);
                }
            }
            if (!f_triangles.isSet() && f_computeTriangles.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("triangles"):topology->findData("triangles");
                if (tparent)
                {
                    f_triangles.setParent(tparent);
                    f_triangles.setReadOnly(true);
                }
            }
            if (!f_tetrahedra.isSet() && f_computeTetrahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("tetrahedra"):topology->findData("tetrahedra");
                if (tparent)
                {
                    f_tetrahedra.setParent(tparent);
                    f_tetrahedra.setReadOnly(true);
                }
            }
            if (!f_hexahedra.isSet() && f_computeHexahedra.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("hexahedra"):topology->findData("hexahedra");
                if (tparent)
                {
                    f_hexahedra.setParent(tparent);
                    f_hexahedra.setReadOnly(true);
                }
            }
            if (!f_quad.isSet() && f_computeQuad.getValue())
            {
                BaseData* tparent = topologyContainer?topologyContainer->findData("quads"):topology->findData("quads");
                if (tparent)
                {
                    f_quad.setParent(tparent);
                    f_quad.setReadOnly(true);
                }
            }
        }/*else{
            msg_warning(this) << "No primitives provided nor TopologyContainer and a BaseMeshTopology in the current context.\n"
                                 "To remove this message you can either: \n"
                                 "  - set value into one or more of the attributes 'edges', 'triangles', 'tetrahedra', 'hexahedra'. \n"
                                 "  - add a TopologyContainer and a BaseMeshTopology in the context of this object. \n";
            m_componentstate = ComponentState::Invalid ;
            return ;
        }*/
    }

    addInput(&f_X0);
    addInput(&f_edges);
    addInput(&f_triangles);
    addInput(&f_tetrahedra);
    addInput(&f_hexahedra);
    addInput(&f_quad);

    addOutput(&f_indices);
    addOutput(&f_edgeIndices);
    addOutput(&f_triangleIndices);
    addOutput(&f_tetrahedronIndices);
    addOutput(&f_hexahedronIndices);
    addOutput(&f_quadIndices);
    addOutput(&f_pointsInROI);
    addOutput(&f_edgesInROI);
    addOutput(&f_trianglesInROI);
    addOutput(&f_tetrahedraInROI);
    addOutput(&f_hexahedraInROI);
    addOutput(&f_quadInROI);
    addOutput(&f_nbIndices);

    m_componentstate = ComponentState::Valid ;

    /// The following is a trick to force the initial selection of the element by the engine.
    bool tmp=p_doUpdate.getValue() ;
    p_doUpdate.setValue(true);
    setDirtyValue();
    reinit();
    p_doUpdate.setValue(tmp);
}

template <class DataTypes>
void BoxROI<DataTypes>::reinit()
{
    vector<Vec6>& vb = *(boxes.beginEdit());
    if (!vb.empty())
    {
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (vb[bi][0] > vb[bi][3]) std::swap(vb[bi][0],vb[bi][3]);
            if (vb[bi][1] > vb[bi][4]) std::swap(vb[bi][1],vb[bi][4]);
            if (vb[bi][2] > vb[bi][5]) std::swap(vb[bi][2],vb[bi][5]);
        }
    }
    boxes.endEdit();

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
    const VecCoord& x0 = f_X0.getValue();
    CPos p =  DataTypes::getCPos(x0[pid]);
    return ( isPointInBox(p,b) );
}

template <class DataTypes>
bool BoxROI<DataTypes>::isEdgeInBox(const Edge& e, const Vec6& b)
{
    const VecCoord& x0 = f_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[e[0]]);
    CPos p1 =  DataTypes::getCPos(x0[e[1]]);
    CPos c = (p1+p0)*0.5;

    return isPointInBox(c,b);
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTriangleInBox(const Triangle& t, const Vec6& b)
{
    const VecCoord& x0 = f_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (isPointInBox(c,b));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isTetrahedronInBox(const Tetra &t, const Vec6 &b)
{
    const VecCoord& x0 = f_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBox(c,b));
}

template <class DataTypes>
bool BoxROI<DataTypes>::isHexahedronInBox(const Hexa &t, const Vec6 &b)
{
    const VecCoord& x0 = f_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[t[0]]);
    CPos p1 =  DataTypes::getCPos(x0[t[1]]);
    CPos p2 =  DataTypes::getCPos(x0[t[2]]);
    CPos p3 =  DataTypes::getCPos(x0[t[3]]);
    CPos p4 =  DataTypes::getCPos(x0[t[4]]);
    CPos p5 =  DataTypes::getCPos(x0[t[5]]);
    CPos p6 =  DataTypes::getCPos(x0[t[6]]);
    CPos p7 =  DataTypes::getCPos(x0[t[7]]);
    CPos c = (p7+p6+p5+p4+p3+p2+p1+p0)/8.0;

    return (isPointInBox(c,b));
}


template <class DataTypes>
bool BoxROI<DataTypes>::isQuadInBox(const Quad& q, const Vec6& b)
{
    const VecCoord& x0 = f_X0.getValue();
    CPos p0 =  DataTypes::getCPos(x0[q[0]]);
    CPos p1 =  DataTypes::getCPos(x0[q[1]]);
    CPos p2 =  DataTypes::getCPos(x0[q[2]]);
    CPos p3 =  DataTypes::getCPos(x0[q[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (isPointInBox(c,b));

}

// The update method is called when the engine is marked as dirty.
template <class DataTypes>
void BoxROI<DataTypes>::update()
{
    if(m_componentstate==ComponentState::Invalid){
        cleanDirty() ;
        return ;
    }

    if(!p_doUpdate.getValue()){
        cleanDirty() ;
        return ;
    }

    const vector<Vec6>& vb = boxes.getValue();

    if (vb.empty()) { cleanDirty(); return; }


    // Read accessor for input topology
    ReadAccessor< Data<vector<Edge> > > edges = f_edges;
    ReadAccessor< Data<vector<Triangle> > > triangles = f_triangles;
    ReadAccessor< Data<vector<Tetra> > > tetrahedra = f_tetrahedra;
    ReadAccessor< Data<vector<Hexa> > > hexahedra = f_hexahedra;
    ReadAccessor< Data<vector<Quad> > > quad = f_quad;

    const VecCoord& x0 = f_X0.getValue();

    cleanDirty();

    // Write accessor for topological element indices in BOX
    SetIndex& indices = *f_indices.beginWriteOnly();
    SetIndex& edgeIndices = *f_edgeIndices.beginWriteOnly();
    SetIndex& triangleIndices = *f_triangleIndices.beginWriteOnly();
    SetIndex& tetrahedronIndices = *f_tetrahedronIndices.beginWriteOnly();
    SetIndex& hexahedronIndices = *f_hexahedronIndices.beginWriteOnly();
    SetIndex& quadIndices = *f_quadIndices.beginWriteOnly();

    // Write accessor for toplogical element in BOX
    WriteOnlyAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
    WriteOnlyAccessor< Data<vector<Edge> > > edgesInROI = f_edgesInROI;
    WriteOnlyAccessor< Data<vector<Triangle> > > trianglesInROI = f_trianglesInROI;
    WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;
    WriteOnlyAccessor< Data<vector<Hexa> > > hexahedraInROI = f_hexahedraInROI;
    WriteOnlyAccessor< Data<vector<Quad> > > quadInROI = f_quadInROI;


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


    //Points
    for( unsigned i=0; i<x0.size(); ++i )
    {
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            if (isPointInBox(i, vb[bi]))
            {
                indices.push_back(i);
                pointsInROI.push_back(x0[i]);
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

    //Hexahedra
    if (f_computeHexahedra.getValue())
    {
        for(unsigned int i=0 ; i<hexahedra.size() ; i++)
        {
            Hexa t = hexahedra[i];
            for (unsigned int bi=0; bi<vb.size(); ++bi)
            {
                if (isHexahedronInBox(t, vb[bi]))
                {
                    hexahedronIndices.push_back(i);
                    hexahedraInROI.push_back(t);
                    break;
                }
            }
        }
    }

    //Quads
    if (f_computeQuad.getValue())
    {
        for(unsigned int i=0 ; i<quad.size() ; i++)
        {
            Quad q = quad[i];
            for (unsigned int bi=0; bi<vb.size(); ++bi)
            {
                if (isQuadInBox(q, vb[bi]))
                {
                    quadIndices.push_back(i);
                    quadInROI.push_back(q);
                    break;
                }
            }
        }
    }


    f_nbIndices.setValue(indices.size());

    f_indices.endEdit();
    f_edgeIndices.endEdit();
    f_triangleIndices.endEdit();
    f_tetrahedronIndices.endEdit();
    f_hexahedronIndices.endEdit();
    f_quadIndices.endEdit();

}


template <class DataTypes>
void BoxROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(m_componentstate==ComponentState::Invalid)
        return ;

    if (!vparams->displayFlags().getShowBehaviorModels() && !this->_drawSize.getValue())
        return;

    const VecCoord& x0 = f_X0.getValue();
    Vec4f color = Vec4f(1.0f, 0.4f, 0.4f, 1.0f);


    ///draw the boxes
    if( p_drawBoxes.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<Vector3> vertices;
        const vector<Vec6>& vb=boxes.getValue();
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            const Vec6& b=vb[bi];
            const Real& Xmin=b[0];
            const Real& Xmax=b[3];
            const Real& Ymin=b[1];
            const Real& Ymax=b[4];
            const Real& Zmin=b[2];
            const Real& Zmax=b[5];
            vertices.push_back( Vector3(Xmin,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmin,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmax) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmax) );
            vertices.push_back( Vector3(Xmin,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmin,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmin,Ymax,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmax) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmax,Ymin,Zmin) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmin) );
            vertices.push_back( Vector3(Xmax,Ymax,Zmax) );
            vparams->drawTool()->drawLines(vertices, linesWidth , color );
        }
    }

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    ///draw points in ROI
    if( p_drawPoints.getValue())
    {
        float pointsWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<Vector3> vertices;
        ReadAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            Vector3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawPoints(vertices, pointsWidth, color);
    }

    ///draw edges in ROI
    if( p_drawEdges.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<Vector3> vertices;
        ReadAccessor< Data<vector<Edge> > > edgesInROI = f_edgesInROI;
        for (unsigned int i=0; i<edgesInROI.size() ; ++i)
        {
            Edge e = edgesInROI[i];
            for (unsigned int j=0 ; j<2 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[e[j]]);
                Vector3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawLines(vertices, linesWidth, color);
    }

    ///draw triangles in ROI
    if( p_drawTriangles.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<Vector3> vertices;
        ReadAccessor< Data<vector<Triangle> > > trianglesInROI = f_trianglesInROI;
        for (unsigned int i=0; i<trianglesInROI.size() ; ++i)
        {
            Triangle t = trianglesInROI[i];
            for (unsigned int j=0 ; j<3 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                Vector3 pv;
                for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                    pv[j] = p[j];
                vertices.push_back( pv );
            }
        }
        vparams->drawTool()->drawTriangles(vertices, color);
    }

    ///draw tetrahedra in ROI
    if( p_drawTetrahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<Vector3> vertices;
        ReadAccessor< Data<vector<Tetra> > > tetrahedraInROI = f_tetrahedraInROI;
        for (unsigned int i=0; i<tetrahedraInROI.size() ; ++i)
        {
            Tetra t = tetrahedraInROI[i];
            for (unsigned int j=0 ; j<4 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                Vector3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            Vector3 pv;
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
    if( p_drawHexahedra.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<Vector3> vertices;
        ReadAccessor< Data<vector<Hexa> > > hexahedraInROI = f_hexahedraInROI;
        for (unsigned int i=0; i<hexahedraInROI.size() ; ++i)
        {
            Hexa t = hexahedraInROI[i];
            for (unsigned int j=0 ; j<8 ; j++)
            {
                CPos p = DataTypes::getCPos(x0[t[j]]);
                Vector3 pv;
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );

                p = DataTypes::getCPos(x0[t[(j+1)%4]]);
                for( unsigned int k=0 ; k<max_spatial_dimensions ; ++k )
                    pv[k] = p[k];
                vertices.push_back( pv );
            }

            CPos p = DataTypes::getCPos(x0[t[0]]);
            Vector3 pv;
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
    if( p_drawQuads.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<Vector3> vertices;
        ReadAccessor<Data<vector<Quad> > > quadsInROI = f_quadInROI;
        for (unsigned i=0; i<quadsInROI.size(); ++i)
        {
            Quad q = quadsInROI[i];
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[j]]);
                Vector3 pv;
                for (unsigned k=0; k<max_spatial_dimensions; k++)
                    pv[k] = p[k];
                vertices.push_back(pv);
            }
            for (unsigned j=0; j<4; j++)
            {
                CPos p = DataTypes::getCPos(x0[q[(j+1)%4]]);
                Vector3 pv;
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
    if( onlyVisible && !p_drawBoxes.getValue() )
        return;

    if(m_componentstate==ComponentState::Invalid)
        return ;

    const vector<Vec6>& vb=boxes.getValue(params);
    const Real max_real = std::numeric_limits<Real>::max();
    const Real min_real = std::numeric_limits<Real>::min();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

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
    this->f_bbox.setValue(params,TBoundingBox<Real>(minBBox,maxBBox));
}


template<class DataTypes>
void BoxROI<DataTypes>::handleEvent(Event *event)
{
    if (AnimateBeginEvent::checkEventType(event))
    {
        setDirtyValue();
        update();
    }
}

} // namespace boxroi

} // namespace engine

} // namespace component

} // namespace sofa

#endif
