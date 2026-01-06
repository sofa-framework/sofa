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
#include <sofa/component/topology/utility/TopologyBoundingTrasher.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::utility
{

using namespace sofa::core::topology;

template <class DataTypes>
TopologyBoundingTrasher<DataTypes>::TopologyBoundingTrasher()
    : d_positions(initData(&d_positions, "position", "position coordinates of the topology object to interact with."))
    , d_borders(initData(&d_borders, Vec6(-1000, -1000, -1000, 1000, 1000, 1000), "box", "List of boxes defined by xmin,ymin,zmin, xmax,ymax,zmax"))
    , d_drawBox(initData(&d_drawBox, false, "drawBox", "Draw bounding box (default = false)"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
    , edgeModifier(nullptr)
    , triangleModifier(nullptr)
    , quadModifier(nullptr)
    , tetraModifier(nullptr)
    , hexaModifier(nullptr)
{
    f_listening.setValue(true);
}

template <class DataTypes>
TopologyBoundingTrasher<DataTypes>::~TopologyBoundingTrasher()
{

}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    
    if (m_topology->getNbHexahedra() > 0)
    {
        m_topologyType = geometry::ElementType::HEXAHEDRON;
        this->getContext()->get(hexaModifier);
        if (!hexaModifier)
        {
            msg_error() << "Hexahedron topology but no Modifier found. Add the component HexahedronSetTopologyModifier.";
            m_topologyType = geometry::ElementType::UNKNOWN;
        }        
    }
    else if (m_topology->getNbTetrahedra() > 0)
    {
        m_topologyType = geometry::ElementType::TETRAHEDRON;
        this->getContext()->get(tetraModifier);
        if (!tetraModifier)
        {
            msg_error() << "Tetrahedron topology but no modifier found. Add the component TetrahedronSetTopologyModifier.";
            m_topologyType = geometry::ElementType::UNKNOWN;
        }        
    }
    else if (m_topology->getNbQuads() > 0)
    {
        m_topologyType = geometry::ElementType::QUAD;
        this->getContext()->get(quadModifier);
        if (!quadModifier)
        {
            msg_error() << "Quad topology but no modifier found. Add the component QuadSetTopologyModifier.";
            m_topologyType = geometry::ElementType::UNKNOWN;
        }        
    }
    else if (m_topology->getNbTriangles() > 0)
    {
        m_topologyType = geometry::ElementType::TRIANGLE;
        this->getContext()->get(triangleModifier);
        if (!triangleModifier)
        {
            msg_error() << "Triangle topology but no modifier found. Add the component TriangleSetTopologyModifier.";
            m_topologyType = geometry::ElementType::UNKNOWN;
        }        
    }
    else if (m_topology->getNbEdges() > 0)
    {
        m_topologyType = geometry::ElementType::EDGE;
        this->getContext()->get(edgeModifier);
        if (!edgeModifier)
        {
            msg_error() << "Edge topology but no modifier found. Add the component EdgeSetTopologyModifier.";
            m_topologyType = geometry::ElementType::UNKNOWN;
        }        
    }

    reinit();
}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::reinit()
{
    const Vec6& border = d_borders.getValue();
    Vec3 minBBox { border[0], border[1], border[2] };
    Vec3 maxBBox { border[3], border[4], border[5] };
    this->f_bbox.setValue(type::BoundingBox(minBBox, maxBBox));
}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::filterElementsToRemove()
{
    SCOPED_TIMER("filterElementsToRemove");

    const VecCoord& positions = d_positions.getValue();
    const Vec6& border = d_borders.getValue();
    m_indicesToRemove.clear();
    int cpt = 0;
    if (m_topologyType == geometry::ElementType::HEXAHEDRON)
    {
        const BaseMeshTopology::SeqHexahedra& hexahedra = m_topology->getHexahedra();
        for (auto& hexa : hexahedra)
        {
            Coord bary;
            for (unsigned int i = 0; i < 8; i++)
                bary += positions[hexa[i]];

            bary /= 8;
            if (isPointOutside(bary, border))
                m_indicesToRemove.push_back(cpt);
            
            cpt++;
        }
    }
    else if (m_topologyType == geometry::ElementType::TETRAHEDRON)
    {
        const BaseMeshTopology::SeqTetrahedra& tetrahedra = m_topology->getTetrahedra();
        for (auto& tetra : tetrahedra)
        {
            Coord bary;
            for (unsigned int i = 0; i < 4; i++)
                bary += positions[tetra[i]];

            bary /= 4;
            if (isPointOutside(bary, border))
                m_indicesToRemove.push_back(cpt);

            cpt++;
        }
    }
    else if (m_topologyType == geometry::ElementType::QUAD)
    {
        const BaseMeshTopology::SeqQuads& quads = m_topology->getQuads();
        for (auto& quad : quads)
        {
            Coord bary;
            for (unsigned int i = 0; i < 4; i++)
                bary += positions[quad[i]];

            bary /= 4;
            if (isPointOutside(bary, border))
                m_indicesToRemove.push_back(cpt);

            cpt++;
        }
    }
    else if (m_topologyType == geometry::ElementType::TRIANGLE)
    {
        const BaseMeshTopology::SeqTriangles& triangles = m_topology->getTriangles();
        for (auto& triangle : triangles)
        {
            Coord bary;
            for (unsigned int i = 0; i < 3; i++)
                bary += positions[triangle[i]];

            bary /= 3;
            if (isPointOutside(bary, border))
                m_indicesToRemove.push_back(cpt);

            cpt++;
        }
    }
    else if (m_topologyType == geometry::ElementType::EDGE)
    {
        const BaseMeshTopology::SeqEdges& edges = m_topology->getEdges();
        for (auto& edge : edges)
        {
            Coord bary;
            for (unsigned int i = 0; i < 2; i++)
                bary += positions[edge[i]];

            bary /= 2;
            if (isPointOutside(bary, border))
                m_indicesToRemove.push_back(cpt);

            cpt++;
        }
    }
   
    if (!m_indicesToRemove.empty())
        removeElements();
}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::removeElements()
{
    if (m_topologyType == geometry::ElementType::HEXAHEDRON)
    {
        hexaModifier->removeHexahedra(m_indicesToRemove);
    }
    else if (m_topologyType == geometry::ElementType::TETRAHEDRON)
    {
        tetraModifier->removeTetrahedra(m_indicesToRemove);
    }
    else if (m_topologyType == geometry::ElementType::QUAD)
    {
        quadModifier->removeQuads(m_indicesToRemove, true, true);
    }
    else if (m_topologyType == geometry::ElementType::TRIANGLE)
    {
        triangleModifier->removeTriangles(m_indicesToRemove, true, true);
    }
    else if (m_topologyType == geometry::ElementType::EDGE)
    {
        edgeModifier->removeEdges(m_indicesToRemove);
    }
    m_indicesToRemove.clear();
}


template <class DataTypes>
constexpr bool TopologyBoundingTrasher<DataTypes>::isPointOutside(const Coord& value, const Vec6& bb)
{
    if (value[0] < bb[0] || value[0] > bb[3]) // check x
        return true;

    if (value[1] < bb[1] || value[1] > bb[4]) // check y
        return true;

    if (value[2] < bb[2] || value[2] > bb[5]) // check z
        return true;

    return false;
}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        filterElementsToRemove();
    }
}


template <class DataTypes>
void TopologyBoundingTrasher<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (d_drawBox.getValue())
    {
        const Vec6& border = d_borders.getValue();
        constexpr auto color = sofa::type::RGBAColor(1.0f, 0.4f, 0.4f, 1.0f);
        std::vector<Vector3> vertices;
        const Real& Xmin = border[0];
        const Real& Xmax = border[3];
        const Real& Ymin = border[1];
        const Real& Ymax = border[4];
        const Real& Zmin = border[2];
        const Real& Zmax = border[5];
        vertices.push_back(Vector3(Xmin, Ymin, Zmin));
        vertices.push_back(Vector3(Xmin, Ymin, Zmax));
        vertices.push_back(Vector3(Xmin, Ymin, Zmin));
        vertices.push_back(Vector3(Xmax, Ymin, Zmin));
        vertices.push_back(Vector3(Xmin, Ymin, Zmin));
        vertices.push_back(Vector3(Xmin, Ymax, Zmin));
        vertices.push_back(Vector3(Xmin, Ymax, Zmin));
        vertices.push_back(Vector3(Xmax, Ymax, Zmin));
        vertices.push_back(Vector3(Xmin, Ymax, Zmin));
        vertices.push_back(Vector3(Xmin, Ymax, Zmax));
        vertices.push_back(Vector3(Xmin, Ymax, Zmax));
        vertices.push_back(Vector3(Xmin, Ymin, Zmax));
        vertices.push_back(Vector3(Xmin, Ymin, Zmax));
        vertices.push_back(Vector3(Xmax, Ymin, Zmax));
        vertices.push_back(Vector3(Xmax, Ymin, Zmax));
        vertices.push_back(Vector3(Xmax, Ymax, Zmax));
        vertices.push_back(Vector3(Xmax, Ymin, Zmax));
        vertices.push_back(Vector3(Xmax, Ymin, Zmin));
        vertices.push_back(Vector3(Xmin, Ymax, Zmax));
        vertices.push_back(Vector3(Xmax, Ymax, Zmax));
        vertices.push_back(Vector3(Xmax, Ymax, Zmin));
        vertices.push_back(Vector3(Xmax, Ymin, Zmin));
        vertices.push_back(Vector3(Xmax, Ymax, Zmin));
        vertices.push_back(Vector3(Xmax, Ymax, Zmax));
        vparams->drawTool()->drawLines(vertices, 1.0, color);
    }
}

} // namespace sofa::component::topology::utility
