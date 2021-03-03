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

#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/fwd.h>

#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYENGINE_DEFINITION
namespace std
{
    extern template class list<const sofa::core::topology::TopologyChange*>;
}
namespace sofa::core::objectmodel
{
    extern template class Data<std::list<const sofa::core::topology::TopologyChange*>>;
}

#endif /// SOFA_CORE_TOPOLOGY_TOPOLOGYENGINE_DEFINITION

namespace sofa
{

namespace core
{

namespace topology
{

/** A class that will interact on a topological Data */
class SOFA_CORE_API TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(TopologyEngine, DataEngine);

protected:
    TopologyEngine() {}

public:
    void init() override ;
    void handleTopologyChange() override {}

public:
    // really need to be a Data??
    Data <std::list<const TopologyChange *> >m_changeList;


    virtual void ApplyTopologyChanges(const std::list< const core::topology::TopologyChange*>& _topologyChangeEvents, const Size _dataSize);

    virtual void ApplyTopologyChange(const core::topology::EndingEvent* /*event*/) {}

    ///////////////////////// Functions on Points //////////////////////////////////////
    /// Apply swap between point indicPointes elements.
    virtual void ApplyTopologyChange(const core::topology::PointsIndicesSwap* /*event*/) {}
    /// Apply adding points elements.
    virtual void ApplyTopologyChange(const core::topology::PointsAdded* /*event*/) {}
    /// Apply removing points elements.
    virtual void ApplyTopologyChange(const core::topology::PointsRemoved* /*event*/) {}
    /// Apply renumbering on points elements.
    virtual void ApplyTopologyChange(const core::topology::PointsRenumbering* /*event*/) {}
    /// Apply moving points elements.
    virtual void ApplyTopologyChange(const core::topology::PointsMoved* /*event*/) {}

    ///////////////////////// Functions on Edges //////////////////////////////////////
    /// Apply swap between edges indices elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesIndicesSwap* /*event*/) {}
    /// Apply adding edges elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesAdded* /*event*/) {}
    /// Apply removing edges elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesRemoved* /*event*/) {}
    /// Apply removing function on moved edges elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesMoved_Removing* /*event*/) {}
    /// Apply adding function on moved edges elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesMoved_Adding* /*event*/) {}
    /// Apply renumbering on edges elements.
    virtual void ApplyTopologyChange(const core::topology::EdgesRenumbering* /*event*/) {}

    ///////////////////////// Functions on Triangles //////////////////////////////////////
    /// Apply swap between triangles indices elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesIndicesSwap* /*event*/) {}
    /// Apply adding triangles elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/) {}
    /// Apply removing triangles elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/) {}
    /// Apply removing function on moved triangles elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesMoved_Removing* /*event*/) {}
    /// Apply adding function on moved triangles elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesMoved_Adding* /*event*/) {}
    /// Apply renumbering on triangles elements.
    virtual void ApplyTopologyChange(const core::topology::TrianglesRenumbering* /*event*/) {}

    ///////////////////////// Functions on Quads //////////////////////////////////////
    /// Apply swap between quads indices elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsIndicesSwap* /*event*/) {}
    /// Apply adding quads elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsAdded* /*event*/) {}
    /// Apply removing quads elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsRemoved* /*event*/) {}
    /// Apply removing function on moved quads elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsMoved_Removing* /*event*/) {}
    /// Apply adding function on moved quads elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsMoved_Adding* /*event*/) {}
    /// Apply renumbering on quads elements.
    virtual void ApplyTopologyChange(const core::topology::QuadsRenumbering* /*event*/) {}

    ///////////////////////// Functions on Tetrahedron //////////////////////////////////////
    /// Apply swap between tetrahedron indices elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraIndicesSwap* /*event*/) {}
    /// Apply adding tetrahedron elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/) {}
    /// Apply removing tetrahedron elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/) {}
    /// Apply removing function on moved tetrahedron elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraMoved_Removing* /*event*/) {}
    /// Apply adding function on moved tetrahedron elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraMoved_Adding* /*event*/) {}
    /// Apply renumbering on tetrahedron elements.
    virtual void ApplyTopologyChange(const core::topology::TetrahedraRenumbering* /*event*/) {}

    ///////////////////////// Functions on Hexahedron //////////////////////////////////////
    /// Apply swap between hexahedron indices elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraIndicesSwap* /*event*/) {}
    /// Apply adding hexahedron elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraAdded* /*event*/) {}
    /// Apply removing hexahedron elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraRemoved* /*event*/) {}
    /// Apply removing function on moved hexahedron elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraMoved_Removing* /*event*/) {}
    /// Apply adding function on moved hexahedron elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraMoved_Adding* /*event*/) {}
    /// Apply renumbering on hexahedron elements.
    virtual void ApplyTopologyChange(const core::topology::HexahedraRenumbering* /*event*/) {}


    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Point>::EMoved_Adding* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Point>::EMoved_Removing* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Edge>::EMoved* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Triangle>::EMoved* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Quad>::EMoved* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Tetrahedron>::EMoved* /*event*/) {}
    virtual void ApplyTopologyChange(const TopologyChangeElementInfo<Topology::Hexahedron>::EMoved* /*event*/) {}



    virtual bool isTopologyDataRegistered() { return false; }


    size_t getNumberOfTopologicalChanges();

    virtual void createEngineName();
    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

    void setNamePrefix(const std::string& s) { m_prefix = s; }

    virtual void registerTopology();
    virtual void registerTopology(sofa::core::topology::BaseMeshTopology* _topology) {}
protected:
    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};

} // namespace topology

} // namespace component

} // namespace sofa
