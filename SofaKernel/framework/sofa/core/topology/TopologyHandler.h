/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H

#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace core
{

namespace topology
{

typedef Topology::Point            Point;
typedef Topology::Edge             Edge;
typedef Topology::Triangle         Triangle;
typedef Topology::Quad             Quad;
typedef Topology::Tetrahedron      Tetrahedron;
typedef Topology::Hexahedron       Hexahedron;


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SOFA_CORE_API TopologyHandler
{
public:
    TopologyHandler() : lastElementIndex(0) {}

    virtual ~TopologyHandler() {}

    virtual void ApplyTopologyChanges(const std::list< const core::topology::TopologyChange *>& _topologyChangeEvents, const unsigned int _dataSize);

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


    virtual bool isTopologyDataRegistered() {return false;}

    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int /*i1*/, unsigned int /*i2*/ ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int> &/*index*/ ) {}

protected:
    /// to handle PointSubsetData
    void setDataSetArraySize(const unsigned int s) { lastElementIndex = s-1; }

    /// to handle properly the removal of items, the container must know the index of the last element
    unsigned int lastElementIndex;
};


} // namespace topology

} // namespace core

} // namespace sofa


#endif // SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H
