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
#include <sofa/component/engine/select/BaseROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <limits>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/helper/accessor.h>

namespace sofa::component::engine::select
{

using core::behavior::BaseMechanicalState ;
using core::topology::BaseMeshTopology ;
using core::objectmodel::ComponentState ;
using core::objectmodel::BaseData ;
using core::objectmodel::Event ;
using core::loader::MeshLoader ;
using core::ExecParams ;
using type::Vec3 ;
using type::Vec4f ;
using helper::WriteOnlyAccessor ;
using helper::ReadAccessor ;
using type::vector ;

template <class DataTypes>
BaseROI<DataTypes>::BaseROI()
    : d_positions( initData (&d_positions, "position", "Rest position coordinates of the degrees of freedom. \n"
                                         "If empty the positions from a MechanicalObject then a MeshLoader are searched in the current context. \n"
                                         "If none are found the parent's context is searched for MechanicalObject." ) )
    , d_edges(initData (&d_edges, "edges", "Edge Topology") )
    , d_triangles(initData (&d_triangles, "triangles", "Triangle Topology") )
    , d_quads(initData (&d_quads, "quads", "Quad Topology") )
    , d_tetrahedra(initData (&d_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , d_hexahedra(initData (&d_hexahedra, "hexahedra", "Hexahedron Topology") )
    , d_computeEdges( initData(&d_computeEdges, true,"computeEdges","If true, will compute edge list and index list inside the ROI.") )
    , d_computeTriangles( initData(&d_computeTriangles, true,"computeTriangles","If true, will compute triangle list and index list inside the ROI.") )
    , d_computeQuads(initData(&d_computeQuads, true, "computeQuads", "If true, will compute quad list and index list inside the ROI."))
    , d_computeTetrahedra( initData(&d_computeTetrahedra, true,"computeTetrahedra","If true, will compute tetrahedra list and index list inside the ROI.") )
    , d_computeHexahedra( initData(&d_computeHexahedra, true,"computeHexahedra","If true, will compute hexahedra list and index list inside the ROI.") )
    , d_strict( initData(&d_strict, true,"strict","If true, an element is inside the box if all of its nodes are inside. If False, only the center point of the element is checked.") )
    , d_indices( initData(&d_indices,"indices","Indices of the points contained in the ROI") )
    , d_edgeIndices( initData(&d_edgeIndices,"edgeIndices","Indices of the edges contained in the ROI") )
    , d_triangleIndices( initData(&d_triangleIndices,"triangleIndices","Indices of the triangles contained in the ROI") )
    , d_quadIndices(initData(&d_quadIndices, "quadIndices", "Indices of the quad contained in the ROI"))
    , d_tetrahedronIndices( initData(&d_tetrahedronIndices,"tetrahedronIndices","Indices of the tetrahedra contained in the ROI") )
    , d_hexahedronIndices( initData(&d_hexahedronIndices,"hexahedronIndices","Indices of the hexahedra contained in the ROI") )
    , d_pointsInROI( initData(&d_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , d_edgesInROI( initData(&d_edgesInROI,"edgesInROI","Edges contained in the ROI") )
    , d_trianglesInROI( initData(&d_trianglesInROI,"trianglesInROI","Triangles contained in the ROI") )
    , d_quadsInROI(initData(&d_quadsInROI, "quadsInROI", "Quad contained in the ROI"))
    , d_tetrahedraInROI( initData(&d_tetrahedraInROI,"tetrahedraInROI","Tetrahedra contained in the ROI") )
    , d_hexahedraInROI( initData(&d_hexahedraInROI,"hexahedraInROI","Hexahedra contained in the ROI") )
    , d_nbIndices( initData(&d_nbIndices,"nbIndices", "Number of selected indices") )
    , d_pointsOutROI(initData(&d_pointsOutROI, "pointsOutROI", "Points not contained in the ROI"))
    , d_edgesOutROI(initData(&d_edgesOutROI, "edgesOutROI", "Edges not contained in the ROI"))
    , d_trianglesOutROI(initData(&d_trianglesOutROI, "trianglesOutROI", "Triangles not contained in the ROI"))
    , d_tetrahedraOutROI(initData(&d_tetrahedraOutROI, "tetrahedraOutROI", "Tetrahedra not contained in the ROI"))
    , d_indicesOut(initData(&d_indicesOut, "indicesOut", "Indices of the points not contained in the ROI"))
    , d_edgeOutIndices(initData(&d_edgeOutIndices, "edgeOutIndices", "Indices of the edges not contained in the ROI"))
    , d_triangleOutIndices(initData(&d_triangleOutIndices, "triangleOutIndices", "Indices of the triangles not contained in the ROI"))
    , d_tetrahedronOutIndices(initData(&d_tetrahedronOutIndices, "tetrahedronOutIndices", "Indices of the tetrahedra not contained in the ROI"))
    , d_drawROI( initData(&d_drawROI,false,"drawROI","Draw the ROI.") )
    , d_drawPoints( initData(&d_drawPoints,false,"drawPoints","Draw Points.") )
    , d_drawEdges( initData(&d_drawEdges,false,"drawEdges","Draw Edges.") )
    , d_drawTriangles( initData(&d_drawTriangles,false,"drawTriangles","Draw Triangles.") )
    , d_drawQuads(initData(&d_drawQuads, false, "drawQuads", "Draw Quads."))
    , d_drawTetrahedra( initData(&d_drawTetrahedra,false,"drawTetrahedra","Draw Tetrahedra.") )
    , d_drawHexahedra( initData(&d_drawHexahedra,false,"drawHexahedra","Draw Tetrahedra.") )
    , d_drawSize(initData(&d_drawSize, 1.0f, "drawSize", "rendering size for ROI and topological elements"))
    , d_doUpdate( initData(&d_doUpdate,(bool)true,"doUpdate","If true, updates the selection at the beginning of simulation steps.") )
{
    sofa::helper::getWriteOnlyAccessor(d_indices).push_back(0);

    addInput(&d_positions);
    addInput(&d_edges);
    addInput(&d_triangles);
    addInput(&d_tetrahedra);
    addInput(&d_hexahedra);
    addInput(&d_quads);

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
    addOutput(&d_quadsInROI);
    addOutput(&d_nbIndices);

    addOutput(&d_pointsOutROI);
    addOutput(&d_edgesOutROI);
    addOutput(&d_trianglesOutROI);
    addOutput(&d_tetrahedraOutROI);
    addOutput(&d_indicesOut);
    addOutput(&d_edgeOutIndices);
    addOutput(&d_triangleOutIndices);
    addOutput(&d_tetrahedronOutIndices);
}

template <class DataTypes>
void BaseROI<DataTypes>::init()
{
    /// If the position attribute is not set we are trying to
    /// automatically load the positions from the current context MechanicalState if any, then
    /// in a MeshLoad if any and in case of failure it will finally search it in the parent's
    /// context.
    if (!d_positions.isSet())
    {
        msg_info(this) << "No attribute 'position' set.\n"
                          "Searching in the context for a MechanicalObject or MeshLoader.\n" ;

        BaseMechanicalState* mstate = nullptr ;
        this->getContext()->get(mstate, core::objectmodel::BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                d_positions.setParent(parent);
                d_positions.setReadOnly(true);
            }
            else
            {
                msg_warning(this) << "No attribute 'rest_position' in component '" << getName() << "'.\n"
                                  << "The BaseROI component thus have no input and is thus deactivated.\n" ;
                d_componentState.setValue(ComponentState::Invalid) ;
                return ;
            }
        }
        else
        {
            MeshLoader* loader = nullptr ;
            this->getContext()->get(loader, core::objectmodel::BaseContext::Local);
            if (loader)
            {
                d_positions.setParent(&loader->d_positions);
                d_positions.setReadOnly(true);
            }
            else   // no local state, no loader => find upward
            {
                this->getContext()->get(mstate, core::objectmodel::BaseContext::SearchUp);
                if(!mstate)
                {
                    msg_error(this) <<  "Unable to find a MechanicalObject for this component. "
                                        "To remove this error message you can either:\n"
                                        "   - to specify the DOF where to apply the BaseROI with the 'position' attribute.\n"
                                        "   - to add MechanicalObject or MeshLoader component before the " << this->getClassName() << " in the scene graph.\n";
                    d_componentState.setValue(ComponentState::Invalid) ;
                    return ;
                }

                BaseData* parent = mstate->findData("rest_position");
                if(!parent)
                {
                    dmsg_error(this) <<  "Unable to find a rest_position attribute in the MechanicalObject '" << mstate->getName() << "'";
                    d_componentState.setValue(ComponentState::Invalid) ;
                    return ;
                }
                d_positions.setParent(parent);
                d_positions.setReadOnly(true);
            }
        }
    }


    if (!d_edges.isSet() || !d_triangles.isSet() || !d_tetrahedra.isSet() || !d_hexahedra.isSet() || !d_quads.isSet() )
    {
        msg_info(this) << "No topology given. Searching for a BaseMeshTopology in the current context.\n";

        BaseMeshTopology* topology = nullptr;
        this->getContext()->get(topology, core::objectmodel::BaseContext::Local);

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
            if (!d_hexahedra.isSet() && d_computeHexahedra.getValue())
            {
                BaseData* tparent = topology->findData("hexahedra");
                if (tparent)
                {
                    d_hexahedra.setParent(tparent);
                    d_hexahedra.setReadOnly(true);
                }
            }
            if (!d_quads.isSet() && d_computeQuads.getValue())
            {
                BaseData* tparent = topology->findData("quads");
                if (tparent)
                {
                    d_quads.setParent(tparent);
                    d_quads.setReadOnly(true);
                }
            }
        }
    }


    d_componentState.setValue(ComponentState::Valid) ;

    /// The following is a trick to force the initial selection of the element by the engine.
    {
        const bool tmp=d_doUpdate.getValue() ;
        d_doUpdate.setValue(true);
        setDirtyValue();
        
        roiInit();
        
        update();
        d_doUpdate.setValue(tmp);
    }
}

template <class DataTypes>
bool BaseROI<DataTypes>::isPointIn(const PointID pid) const
{
    const VecCoord& positions = d_positions.getValue();
    const CPos& p = DataTypes::getCPos(positions[pid]);
    return isPointInROI(p);
}

// The update method is called when the engine is marked as dirty.
template <class DataTypes>
void BaseROI<DataTypes>::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid)
    {
        return ;
    }


    if(d_doUpdate.getValue())
    {

        // Check whether an element can partially be inside the box or if all of its nodes must be inside
        const bool strict = d_strict.getValue();

        // Write accessor for topological element indices in BOX
        SetIndex& indices = *d_indices.beginWriteOnly();
        SetIndex& edgeIndices = *d_edgeIndices.beginWriteOnly();
        SetIndex& triangleIndices = *d_triangleIndices.beginWriteOnly();
        SetIndex& quadIndices = *d_quadIndices.beginWriteOnly();
        SetIndex& tetrahedronIndices = *d_tetrahedronIndices.beginWriteOnly();
        SetIndex& hexahedronIndices = *d_hexahedronIndices.beginWriteOnly();
        SetIndex& indicesOut = *d_indicesOut.beginWriteOnly();
        SetIndex& edgeOutIndices = *d_edgeOutIndices.beginWriteOnly();
        SetIndex& triangleOutIndices = *d_triangleOutIndices.beginWriteOnly();
        SetIndex& quadOutIndices = *d_quadOutIndices.beginWriteOnly();
        SetIndex& tetrahedronOutIndices = *d_tetrahedronOutIndices.beginWriteOnly();
        SetIndex& hexahedronOutIndices = *d_hexahedronOutIndices.beginWriteOnly();

        // Write accessor for toplogical element in BOX
        WriteOnlyAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
        WriteOnlyAccessor< Data<vector<Edge> > > edgesInROI = d_edgesInROI;
        WriteOnlyAccessor< Data<vector<Triangle> > > trianglesInROI = d_trianglesInROI;
        WriteOnlyAccessor< Data<vector<Quad> > > quadInROI = d_quadsInROI;
        WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
        WriteOnlyAccessor< Data<vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
        WriteOnlyAccessor< Data<VecCoord > > pointsOutROI = d_pointsOutROI;
        WriteOnlyAccessor< Data<vector<Edge> > > edgesOutROI = d_edgesOutROI;
        WriteOnlyAccessor< Data<vector<Triangle> > > trianglesOutROI = d_trianglesOutROI;
        WriteOnlyAccessor< Data<vector<Quad> > > quadsOutROI = d_quadsOutROI;
        WriteOnlyAccessor< Data<vector<Tetra> > > tetrahedraOutROI = d_tetrahedraOutROI;
        WriteOnlyAccessor< Data<vector<Hexa> > > hexahedraOutROI = d_hexahedraOutROI;

        // Clear lists
        indices.clear();
        edgeIndices.clear();
        triangleIndices.clear();
        quadIndices.clear();
        tetrahedronIndices.clear();
        hexahedronIndices.clear();
        indicesOut.clear();
        edgeOutIndices.clear();
        triangleOutIndices.clear();
        quadOutIndices.clear();
        tetrahedronOutIndices.clear();
        hexahedronOutIndices.clear();

        pointsInROI.clear();
        edgesInROI.clear();
        trianglesInROI.clear();
        quadInROI.clear();
        tetrahedraInROI.clear();
        hexahedraInROI.clear();
        pointsOutROI.clear();
        edgesOutROI.clear();
        trianglesOutROI.clear();
        quadsOutROI.clear();
        tetrahedraOutROI.clear();
        hexahedraOutROI.clear();


        if (d_positions.getValue().size() == 0)
        {
            msg_warning() << "No rest position yet defined. ROI might not work properly. \n"
                            "This may be caused by an early initialization of the ROI before  \n"
                            "the mesh or the MechanicalObject of the node was initialized too";
            return;
        }

        if (!roiDoUpdate())
        {
            return;
        }

        // Read accessor for input topology
        const ReadAccessor< Data<vector<Edge> > > edges = d_edges;
        const ReadAccessor< Data<vector<Triangle> > > triangles = d_triangles;
        const ReadAccessor< Data<vector<Tetra> > > tetrahedra = d_tetrahedra;
        const ReadAccessor< Data<vector<Hexa> > > hexahedra = d_hexahedra;
        const ReadAccessor< Data<vector<Quad> > > quads = d_quads;

        const VecCoord& positions = d_positions.getValue();

        //Points
        for( unsigned i=0; i<positions.size(); ++i )
        {
            if (isPointIn(i))
            {
                indices.push_back(i);
                pointsInROI.push_back(positions[i]);
            }
            else
            {
                indicesOut.push_back(i);
                pointsOutROI.push_back(positions[i]);
            }
        }

        auto testROI = [&](const auto& elements, const auto& predicate, const auto& strictPredicate, bool strict, 
            auto& inIndices, auto& inROI, auto& outIndices, auto& outROI)
            {
                for (std::size_t i = 0; i < elements.size(); i++)
                {
                    const auto& e = elements[i];
                    const bool isInROI = (strict) ? strictPredicate(e) : predicate(e);
                    if (isInROI)
                    {
                        inIndices.push_back(static_cast<sofa::Index>(i));
                        inROI.push_back(e);
                    }
                    else
                    {
                        outIndices.push_back(static_cast<sofa::Index>(i));
                        outROI.push_back(e);
                    }
                }
            };

        //Edges
        if (d_computeEdges.getValue())
        {
            testROI(edges, [this](auto&& x) {return isEdgeInROI(std::forward<decltype(x)>(x));}, 
                           [this](auto&& x) {return isEdgeInStrictROI(std::forward<decltype(x)>(x)); }, 
                    strict, edgeIndices, edgesInROI, edgeOutIndices, edgesOutROI);
        }

        //Triangles
        if (d_computeTriangles.getValue())
        {
            testROI(triangles, [this](auto&& x) {return isTriangleInROI(std::forward<decltype(x)>(x)); },
                [this](auto&& x) {return isTriangleInStrictROI(std::forward<decltype(x)>(x)); },
                strict, triangleIndices, trianglesInROI, triangleOutIndices, trianglesOutROI);
        }

        //Quads
        if (d_computeQuads.getValue())
        {
            testROI(quads, [this](auto&& x) {return isQuadInROI(std::forward<decltype(x)>(x)); },
                [this](auto&& x) {return isQuadInStrictROI(std::forward<decltype(x)>(x)); },
                strict, quadIndices, quadInROI, quadOutIndices, quadsOutROI);

        }

        //Tetrahedra
        if (d_computeTetrahedra.getValue())
        {
            testROI(tetrahedra, [this](auto&& x) {return isTetrahedronInROI(std::forward<decltype(x)>(x)); },
                [this](auto&& x) {return isTetrahedronInStrictROI(std::forward<decltype(x)>(x)); },
                strict, tetrahedronIndices, tetrahedraInROI, tetrahedronOutIndices, tetrahedraOutROI);
        }

        //Hexahedra
        if (d_computeHexahedra.getValue())
        {
            testROI(hexahedra, [this](auto&& x) {return isHexahedronInROI(std::forward<decltype(x)>(x)); },
                [this](auto&& x) {return isHexahedronInStrictROI(std::forward<decltype(x)>(x)); },
                strict, hexahedronIndices, hexahedraInROI, hexahedronOutIndices, hexahedraOutROI);
        }

        d_nbIndices.setValue(sofa::Size(indices.size()));
    }
}


template <class DataTypes>
void BaseROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return ;

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const VecCoord& positions = d_positions.getValue();
    constexpr auto color = sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (d_drawROI.getValue())
    {
        roiDraw(vparams);
    }

    const float sizeFactor = std::max(d_drawSize.getValue(), 1.0f);

    vparams->drawTool()->setLightingEnabled(false);

    constexpr auto convertToVertex = [](const auto& coord) -> type::Vec3 {

        if constexpr(DataTypes::spatial_dimensions == 1)
        {
            return { coord[0], 0.0, 0.0};
        }
        if constexpr (DataTypes::spatial_dimensions == 2)
        {
            return { coord[0], coord[1], 0.0 };
        }
        if constexpr (DataTypes::spatial_dimensions >= 3)
        {
            return { coord[0], coord[1], coord[2] };
        }

    };

    ///draw points in ROI
    if( d_drawPoints.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<VecCoord > > pointsInROI = d_pointsInROI;
        vertices.reserve(pointsInROI.size());
        for (const auto& p : pointsInROI)
        {
            vertices.push_back(convertToVertex(DataTypes::getCPos(p)));
        }
        vparams->drawTool()->drawPoints(vertices, sizeFactor, color);
    }

    ///draw edges in ROI
    if( d_drawEdges.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Edge> > > edgesInROI = d_edgesInROI;
        vertices.reserve(edgesInROI.size() * 2);
        for (const auto& e : edgesInROI)
        {
            for(const auto eid : e)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[eid])));
            }
        }
        vparams->drawTool()->drawLines(vertices, sizeFactor, color);
    }

    ///draw triangles in ROI
    if( d_drawTriangles.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Triangle> > > trianglesInROI = d_trianglesInROI;
        vertices.reserve(trianglesInROI.size() * 3);
        for (const auto& t : trianglesInROI)
        {
            for(const auto tid : t)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[tid])));
            }
        }
        vparams->drawTool()->drawTriangles(vertices, color);
    }

    ///draw quads in ROI
    if (d_drawQuads.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor<Data<vector<Quad> > > quadsInROI = d_quadsInROI;
        vertices.reserve(quadsInROI.size() * 8);
        for (const auto& q : quadsInROI)
        {
            for (unsigned j = 0; j < 4; j++)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[q[j]])));
            }
            for (unsigned j = 0; j < 4; j++)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[q[(j + 1) % 4]])));
            }

        }
        vparams->drawTool()->drawLines(vertices, sizeFactor, color);
    }

    ///draw tetrahedra in ROI
    if( d_drawTetrahedra.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Tetra> > > tetrahedraInROI = d_tetrahedraInROI;
        vertices.reserve(tetrahedraInROI.size() * 12);
        for (const auto& t : tetrahedraInROI)
        {
            for (unsigned int j=0 ; j<4 ; j++)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[j]])));
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[(j + 1) % 4]])));
            }

            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[0]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[2]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[1]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[t[3]])));
        }
        vparams->drawTool()->drawLines(vertices, sizeFactor, color);
    }

    ///draw hexahedra in ROI
    if( d_drawHexahedra.getValue())
    {
        std::vector<type::Vec3> vertices;
        ReadAccessor< Data<vector<Hexa> > > hexahedraInROI = d_hexahedraInROI;
        vertices.reserve(hexahedraInROI.size() * 24);
        for (const auto& h : hexahedraInROI)
        {
            for (unsigned int j=0 ; j<8 ; j++)
            {
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[j]])));
                vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[(j + 1) % 4]])));
            }

            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[0]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[2]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[1]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[3]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[4]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[5]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[6]])));
            vertices.push_back(convertToVertex(DataTypes::getCPos(positions[h[7]])));
        }
        vparams->drawTool()->drawLines(vertices, sizeFactor, color);
    }

}

template<typename DataTypes>
void BaseROI<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (onlyVisible && !this->d_drawROI.getValue())
        return;

    if (this->d_componentState.getValue() == ComponentState::Invalid)
        return;

    sofa::type::BoundingBox bbox{};
    roiComputeBBox(params, bbox);

    // add the ROI of the input vertices ?

    this->f_bbox.setValue(bbox);
}

template<typename DataTypes, typename Element>
constexpr auto getCenter(const Element& e, const typename DataTypes::VecCoord & positions) -> typename DataTypes::CPos
{
    constexpr auto NumberOfNodes = Element::NumberOfNodes;

    assert(NumberOfNodes > 0);

    typename DataTypes::CPos center{};
    for (const auto eid : e)
    {
        center += DataTypes::getCPos(positions[eid]);
    }

    center = center / static_cast<typename DataTypes::Real>(NumberOfNodes);

    return center;
}

template<typename DataTypes, typename Element>
bool isElementInROI(const Element& e, const typename DataTypes::VecCoord& positions, const std::function<bool(const typename DataTypes::CPos&)>& isPointInROI)
{
    const auto center = getCenter<DataTypes>(e, positions);

    return isPointInROI(center);
}

template<typename DataTypes, typename Element>
bool isElementInStrictROI(const Element& e, const typename DataTypes::VecCoord& positions, const std::function<bool(const typename DataTypes::CPos&)>& isPointInROI)
{
    return std::all_of(e.cbegin(), e.cend(), 
        [&](const auto eid) { return isPointInROI(DataTypes::getCPos(positions[eid])); });
}

template <class DataTypes>
template <typename Element>
bool BaseROI<DataTypes>::isInROI(const Element& e) const
{
    const VecCoord& positions = d_positions.getValue();

    return isElementInROI<DataTypes, Element>(e, positions, [this](auto&& x) {
        return isPointInROI(std::forward<decltype(x)>(x));
        }) ;
}

template <class DataTypes>
template <typename Element>
bool BaseROI<DataTypes>::isInStrictROI(const Element& e) const
{
    const VecCoord& positions = d_positions.getValue();

    return isElementInStrictROI<DataTypes, Element>(e, positions, [this](auto&& x) {
        return isPointInROI(std::forward<decltype(x)>(x));
        });
}


template <class DataTypes>
bool BaseROI<DataTypes>::isEdgeInROI(const Edge& e) const
{
    return isInROI(e);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isEdgeInStrictROI(const Edge& e) const
{
    return isInStrictROI(e);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isTriangleInROI(const Triangle& t) const
{
    return isInROI(t);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isTriangleInStrictROI(const Triangle& t) const
{
    return isInStrictROI(t);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isQuadInROI(const Quad& q) const
{
    return isInROI(q);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isQuadInStrictROI(const Quad& q) const
{
    return isInStrictROI(q);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isTetrahedronInROI(const Tetra& t) const
{
    return isInROI(t);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isTetrahedronInStrictROI(const Tetra& t) const
{
    return isInStrictROI(t);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isHexahedronInROI(const Hexa& h) const
{
    return isInROI(h);
}

template <class DataTypes>
bool BaseROI<DataTypes>::isHexahedronInStrictROI(const Hexa& t) const
{
    return isInStrictROI(t);
}


} // namespace sofa::component::engine::selectI
