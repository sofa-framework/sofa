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
#include <SofaBaseTopology/config.h>

#include <SofaBaseTopology/TopologyEngine.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseTopology/TopologySubsetDataHandler.h>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing element related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: elements added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologySubsetData : public sofa::component::topology::TopologyData<TopologyElementType, VecT>
{

public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// Size
    typedef typename container_type::Size Size;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;


    /// Constructor
    TopologySubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : sofa::component::topology::TopologyData< TopologyElementType, VecT >(data)
        , m_topologyHandler(nullptr)
    {}


    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology, sofa::core::topology::TopologyHandler* _topologyHandler);

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);


protected:
    sofa::component::topology::TopologySubsetDataHandler<TopologyElementType,VecT>* m_topologyHandler;

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Element Topology Data Implementation   ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT > using PointSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Point, VecT>;
template< class VecT > using EdgeSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Edge, VecT>;
template< class VecT > using TriangleSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Triangle, VecT>;
template< class VecT > using QuadSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Quad, VecT>;
template< class VecT > using TetrahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Tetrahedron, VecT>;
template< class VecT > using HexahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Hexahedron, VecT>;

} //namespace sofa::component::topology
